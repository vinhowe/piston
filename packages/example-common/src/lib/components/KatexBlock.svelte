<script lang="ts">
  import katex from "katex";

  import "katex/dist/katex.min.css";

  let { text }: { text?: string } = $props();

  const latexRegex = /\$(.*?)\$/gs;

  function getProcessedContent(originalText: string): { output: string; isHtml: boolean } {
    // Reset lastIndex before using test, as latexRegex is a global regex with state.
    latexRegex.lastIndex = 0;
    if (!latexRegex.test(originalText) || typeof document === "undefined") {
      // No LaTeX found, return original text, mark as not HTML.
      return { output: originalText, isHtml: false };
    }

    // Reset lastIndex again before using replace, to ensure it processes from the start.
    latexRegex.lastIndex = 0;
    const htmlOutput = originalText.replace(latexRegex, (match, latex) => {
      try {
        const rendered = katex.renderToString(latex, {
          throwOnError: false,
        });
        // This allows us to use katex blocks in something that wraps the surrounding text in uppercase.
        return `<span class="normal-case">${rendered}</span>`;
      } catch (e) {
        console.error("Error rendering KaTeX:", e);
        return `<span style="color: red;">${match}</span>`;
      }
    });
    return { output: htmlOutput, isHtml: true };
  }

  const processed = $derived(text ? getProcessedContent(text) : { output: "", isHtml: false });
</script>

{#if processed.isHtml}
  <!-- eslint-disable-next-line svelte/no-at-html-tags -->
  {@html processed.output}
{:else}
  {processed.output}
{/if}
