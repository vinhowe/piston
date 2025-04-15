import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import { fileURLToPath } from 'url';
import path from 'path';

// Get the project root directory
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

export default defineConfig({
	plugins: [sveltekit(), tailwindcss(), wasm(), topLevelAwait()],
	worker: {
		format: 'es',
		plugins: () => [wasm(), topLevelAwait()]
	},
	esbuild: {
		supported: {
			'top-level-await': true
		}
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
		}
	}
});
