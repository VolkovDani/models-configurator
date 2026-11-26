import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontend = fileURLToPath(new URL('../', import.meta.url));
const modules = join(frontend, 'node_modules');
const assets = resolve(frontend, '../internal/status/chart');
const staging = mkdtempSync(join(tmpdir(), 'models-chart-'));
const bundleName = 'history-chart.js';
const licensesName = 'licenses.txt';
const checking = process.argv.includes('--check');

try {
  const bundle = join(staging, bundleName);
  const metafile = join(staging, 'metafile.json');
  execFileSync(
    'bun',
    [
      'build',
      './src/chart.tsx',
      '--target=browser',
      '--format=esm',
      '--production',
      '--banner=/*! Third-party licenses: /history-chart-licenses.txt */',
      `--outfile=${bundle}`,
      `--metafile=${metafile}`,
    ],
    { cwd: frontend, stdio: 'inherit' },
  );
  const { outputs } = JSON.parse(readFileSync(metafile, 'utf8'));
  const [output] = Object.values(outputs);
  if (Object.keys(outputs).length !== 1 || output.imports.length !== 0) {
    throw new Error('Expected one self-contained browser bundle');
  }
  const packages = new Map();
  for (const input of Object.keys(output.inputs)) {
    let dir = dirname(resolve(frontend, input));
    while (dir.startsWith(modules + sep)) {
      const manifest = join(dir, 'package.json');
      if (existsSync(manifest)) {
        const { name, version } = JSON.parse(readFileSync(manifest, 'utf8'));
        if (name && version) {
          packages.set(`${name}@${version}`, dir);
          break;
        }
      }
      dir = dirname(dir);
    }
  }
  // Ship the notices for all dependency code included in the browser build.
  const notices = Array.from(packages.keys())
    .sort()
    .map((pkg) => {
      const dir = packages.get(pkg);
      const names = readdirSync(dir).filter((name) =>
        /^(licen[cs]e|notice|copying)([-.].*)?$/i.test(name),
      );
      if (names.length === 0) throw new Error(`Missing license notice for ${pkg}`);
      const texts = names
        .sort()
        .map((name) => readFileSync(join(dir, name), 'utf8').replaceAll('\r\n', '\n').trimEnd());
      return `${pkg}\n\n${texts.join('\n\n')}`;
    });
  const licenseText = `Third-party licenses for the model history chart\n\n${notices.join('\n\n---\n\n')}\n`;

  for (const [name, contents] of [
    [bundleName, readFileSync(bundle)],
    [licensesName, Buffer.from(licenseText)],
  ]) {
    const path = join(assets, name);
    if (checking) {
      if (!existsSync(path) || !contents.equals(readFileSync(path))) {
        throw new Error(`${name} is out of date. Run bun run build in frontend/.`);
      }
    } else {
      writeFileSync(path, contents);
    }
  }
} finally {
  rmSync(staging, { recursive: true, force: true });
}
