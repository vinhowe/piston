import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import fs from 'node:fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

// Get the project root directory
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

// Remove large static subfolders we don't want to ship
const pruneStaticDirs = () => {
	return {
		name: 'prune-static-dirs',
		// After Vite writes bundles (client/server into .svelte-kit/output/*),
		// remove unwanted static folders from client output to avoid copying into final build
		async writeBundle() {
			const root = path.dirname(fileURLToPath(import.meta.url));
			const clientOut = path.resolve(root, '.svelte-kit', 'output', 'client');
			const targets = [path.resolve(clientOut, 'tokenized'), path.resolve(clientOut, 'tokenizer')];
			await Promise.all(
				targets.map((p) => fs.rm(p, { recursive: true, force: true }).catch(() => {}))
			);
		}
	};
};

export default defineConfig((_) => ({
	plugins: [tailwindcss(), sveltekit(), wasm(), pruneStaticDirs()],
	worker: {
		format: 'es',
		plugins: () => [wasm(), sveltekit()]
	},
	resolve: {
		dedupe: [
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
		supported: { 'top-level-await': true }
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
}));
