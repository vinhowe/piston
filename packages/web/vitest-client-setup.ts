import { vi } from "vitest";

// For browser environments only
if (typeof window !== 'undefined') {
  // required for svelte5 + jsdom as jsdom does not support matchMedia
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    enumerable: true,
    value: vi.fn().mockImplementation((query) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}