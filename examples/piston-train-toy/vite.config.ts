import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

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
	}
});
