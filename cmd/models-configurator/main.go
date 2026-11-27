package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/VolkovDani/models-configurator/internal/apidocs"
	"github.com/VolkovDani/models-configurator/internal/history"
	"github.com/VolkovDani/models-configurator/internal/manager"
	"github.com/VolkovDani/models-configurator/internal/status"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "history" {
		if err := historyCommand(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	config := flag.String("config", "models.json", "path to the model configuration")
	listen := flag.String("listen", "127.0.0.1:8317", "HTTP listen address")
	usage := flag.Usage
	flag.Usage = func() {
		usage()
		fmt.Fprintln(flag.CommandLine.Output(), "\n  history clear\n    Back up and clear model call history while the server is stopped")
	}
	flag.Parse()
	cfg, err := manager.LoadConfig(*config)
	if err != nil {
		log.Fatal(err)
	}
	release, err := history.Lock(historyPath())
	if err != nil {
		log.Fatal(err)
	}
	defer release()
	handler, err := service(cfg)
	if err != nil {
		log.Fatal(err)
	}
	srv := &http.Server{
		Addr:              *listen,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       20 * time.Second,
		WriteTimeout:      150 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    64 << 10,
	}
	log.Printf("Models Configurator listening on %s", *listen)
	log.Fatal(srv.ListenAndServe())
}

func historyPath() string {
	if path := os.Getenv("MODELS_HISTORY_PATH"); path != "" {
		return path
	}
	return "logs/model-history.jsonl"
}

func historyCommand(args []string) error {
	flags := flag.NewFlagSet("history clear", flag.ContinueOnError)
	flags.Usage = func() {
		fmt.Fprintln(flags.Output(), "Usage: models-configurator history clear\nBack up and clear model call history while the server is stopped.")
	}
	if len(args) == 0 || (len(args) == 1 && (args[0] == "-h" || args[0] == "--help")) {
		flags.Usage()
		return nil
	}
	if args[0] != "clear" {
		return errors.New("expected 'history clear'")
	}
	if err := flags.Parse(args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("'history clear' takes no arguments")
	}
	backup, err := history.Clear(historyPath())
	if err != nil {
		return err
	}
	if backup == "" {
		fmt.Println("Model call history is already empty.")
	} else {
		fmt.Printf("Cleared model call history. Backup saved at %s\n", backup)
	}
	return nil
}

func parallelRequestLimit() (int, error) {
	value, set := os.LookupEnv("MODELS_MAX_PARALLEL_REQUESTS")
	if !set {
		return 1, nil
	}
	limit, err := strconv.Atoi(value)
	if err != nil || limit < 1 {
		return 0, errors.New("MODELS_MAX_PARALLEL_REQUESTS must be a positive integer")
	}
	return limit, nil
}

func service(cfg manager.Config) (http.Handler, error) {
	limit, err := parallelRequestLimit()
	if err != nil {
		return nil, err
	}
	settings := cfg.Settings(limit)
	active := cfg
	if path := os.Getenv("MODELS_SETTINGS_PATH"); path != "" {
		settings, err = manager.ReadSettings(path)
		if err != nil {
			return nil, fmt.Errorf("read model settings: %w", err)
		}
		active, err = settings.Apply(cfg)
		if err != nil {
			return nil, fmt.Errorf("apply model settings: %w", err)
		}
	}
	calls, err := history.Open(historyPath())
	if err != nil {
		return nil, fmt.Errorf("open model call history: %w", err)
	}
	api, err := manager.NewWithHistory(active, settings.MaxParallel, func(call history.Call) {
		if err := calls.Record(call); err != nil {
			log.Printf("Record model call history: %v", err)
		}
	})
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	mux.Handle("/v1/", api)
	apidocs.Register(mux)
	status.RegisterTheme(mux)
	status.RegisterHistory(mux, calls)
	status.RegisterSettings(mux, cfg, settings)
	mux.Handle("/", status.New(active))
	return localAccess(mux), nil
}
