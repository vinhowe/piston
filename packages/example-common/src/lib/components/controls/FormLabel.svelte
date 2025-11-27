<script lang="ts">
  import KatexBlock from "../KatexBlock.svelte";
  import { twMerge } from "tailwind-merge";

  type $$Props = {
    forInputId?: string;
    class?: string;
    value: string;
  };
  let { forInputId, class: additionalClasses = "", value }: $$Props = $props();
</script>

<label
  for={forInputId}
  class={twMerge("block text-neutral-700 text-sm font-medium", additionalClasses)}
>
  <KatexBlock text={value} />
</label>
