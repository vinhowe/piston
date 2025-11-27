import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	build: {
		rollupOptions: {
			// Make sure to externalize Svelte so it doesn't get bundled twice
			external: ['svelte', 'svelte/internal'],
			output: {
				globals: {
					svelte: 'Svelte',
				},
			},
		},
	}
});
