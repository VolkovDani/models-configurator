package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/VolkovDani/models-configurator/internal/manager"
	"github.com/VolkovDani/models-configurator/internal/status"
)

func main() {
	config := flag.String("config", "models.json", "path to the model configuration")
	listen := flag.String("listen", "127.0.0.1:8317", "HTTP listen address")
	flag.Parse()
	cfg, err := manager.LoadConfig(*config)
	if err != nil {
		log.Fatal(err)
	}
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

func service(cfg manager.Config) (http.Handler, error) {
	api, err := manager.New(cfg)
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	mux.Handle("/v1/", api)
	mux.Handle("/", status.New(cfg))
	return localAccess(mux), nil
}
