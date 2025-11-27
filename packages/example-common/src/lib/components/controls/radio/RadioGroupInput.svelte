<script lang="ts">
  import FormLabel from "../FormLabel.svelte";
  import ResetValueButton from "../ResetValueButton.svelte";
  import RadioInput from "./RadioInput.svelte";
  import { twMerge } from "tailwind-merge";

  type Option = { value: string; label?: string; disabled?: boolean };

  type $$Props = {
    label?: string;
    name: string;
    value: string;
    options: Option[];
    id: string;
    class?: string;
    itemClass?: string;
    hasDefaultValue?: boolean;
    onReset?: () => void;
    direction?: "row" | "col";
  };

  let {
    label,
    name,
    value = $bindable(),
    options = [],
    id,
    class: wrapperClass = "",
    itemClass = "",
    hasDefaultValue = false,
    onReset = undefined,
    direction = "row",
  }: $$Props = $props();
</script>

<div class={twMerge("relative flex flex-col gap-0.5", wrapperClass)}>
  {#if label}
    <FormLabel forInputId={id} value={label} />
  {/if}
  <div class="flex gap-2 items-start">
    <div
      {id}
      class="flex {direction === 'row' ? 'flex-row gap-1.5 flex-wrap' : 'flex-col gap-1'}"
      role="radiogroup"
    >
      {#each options as opt (String(opt.value))}
        <div class={itemClass}>
          <RadioInput
            {name}
            bind:group={value}
            value={String(opt.value)}
            id={`${id}-${String(opt.value)}`}
            disabled={opt.disabled}
            label={opt.label ?? String(opt.value)}
          />
        </div>
      {/each}
    </div>
    {#if onReset}
      <div class="self-center">
        <ResetValueButton {hasDefaultValue} {onReset} />
      </div>
    {/if}
  </div>
</div>
