import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { execSync } from 'node:child_process';
import path from 'path';
import sirv from 'sirv';
import { fileURLToPath } from 'url';
import { defineConfig, loadEnv, type ViteDevServer } from 'vite';
import wasm from 'vite-plugin-wasm';

// Get the project root directory
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

// Dev-only mount for tokenizer and tokenized directories via env paths
const devStaticMount = (opts: { tokenizerDir?: string; tokenizedDir?: string }) => ({
	name: 'dev-static-mount',
	apply: 'serve',
	configureServer(server: ViteDevServer) {
		const tokenizerDir = opts.tokenizerDir;
		const tokenizedDir = opts.tokenizedDir;
		if (tokenizerDir)
			server.middlewares.use('/tokenizer', sirv(tokenizerDir, { dev: true, etag: true }));
		if (tokenizedDir)
			server.middlewares.use('/tokenized', sirv(tokenizedDir, { dev: true, etag: true }));
	}
});

const commitHash = execSync('git rev-parse --short HEAD').toString().trim();

export default defineConfig(({ mode }) => {
	const envDir = path.dirname(fileURLToPath(import.meta.url));
	const env = loadEnv(mode, envDir, '');

	return {
		define: {
			__COMMIT_HASH__: JSON.stringify(commitHash)
		},
		plugins: [
			tailwindcss(),
			...(mode === 'development'
				? [
						devStaticMount({
							tokenizerDir: env.VITE_TOKENIZER_DIR,
							tokenizedDir: env.VITE_TOKENIZED_DIR
						})
					]
				: []),
			sveltekit(),
			wasm()
		],
		worker: {
			format: 'es',
			plugins: () => [wasm(), sveltekit()]
		},
		resolve: {
			dedupe: [
				'svelte',
				'svelte/legacy',
				'@codemirror/state',
				'@codemirror/view',
				'@codemirror/language',
				'@codemirror/lang-javascript',
				'@codemirror/lint',
				'codemirror',
				'@lezer/highlight'
			]
		},
		esbuild: {
			supported: { 'top-level-await': true },
			keepNames: true
		},
		server: {
			fs: {
				// Allow serving files from the project root and one level up
				allow: [
					// Allow serving from the Svelte project directory
					path.resolve(path.dirname(fileURLToPath(import.meta.url))),
					// Allow serving from the entire ratchet project directory
					projectRoot,
					// Allow serving from the WASM file's directory
					path.resolve(projectRoot, 'target', 'pkg', 'piston-web')
				]
			},
			headers: {
				'Cross-Origin-Embedder-Policy': 'require-corp',
				'Cross-Origin-Opener-Policy': 'same-origin'
			}
		}
	};
});
