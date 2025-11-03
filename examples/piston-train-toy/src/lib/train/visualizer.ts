import { type IRectangle, MaxRectsPacker, PACKING_LOGIC } from 'maxrects-packer';

import { type CaptureMatch } from './capture';

export type MatchBox = {
	matchId: number;
	x: number;
	y: number;
	width: number;
	height: number;
	tileWidth: number;
	tileHeight: number;
	gridRows: number;
	gridCols: number;
	match: CaptureMatch;
};

export type MatchStats = { mean: number | null; variance: number | null } | null;

type PreparedMatch = {
	match: CaptureMatch & { buffer: GPUBuffer };
	gridRows: number;
	gridCols: number;
	tileWidth: number;
	tileHeight: number;
	width: number;
	height: number;
	scale: number;
};

type PackedRect = IRectangle & { id?: number; matchId: number };

const PADDING_PX = 4;
const GLOBAL_PADDING_PX = 3;

export class Visualizer {
	private device: GPUDevice;
	private context: GPUCanvasContext | null = null;
	private canvas: OffscreenCanvas | null = null;
	private canvasFormat: GPUTextureFormat = 'bgra8unorm';
	private storageFormat: GPUTextureFormat = 'rgba8unorm';
	private composeTexture: GPUTexture | null = null;
	private composeView: GPUTextureView | null = null;
	private blitPipeline: GPURenderPipeline | null = null;
	private computePipeline: GPUComputePipeline | null = null;
	private reducePipelineStage1: GPUComputePipeline | null = null;
	private reducePipelineStage2: GPUComputePipeline | null = null;
	private blitBindGroupLayout: GPUBindGroupLayout | null = null;
	private blitSampler: GPUSampler | null = null;
	private cssLabelPaddingPx: number = 0;
	private targetWidth: number = 1;
	private targetHeight: number = 1;
	private maxTextureDim: number = 8192;
	private needsResize: boolean = false;

	constructor(device: GPUDevice) {
		this.device = device;
	}

	init(canvas: OffscreenCanvas, format?: GPUTextureFormat) {
		this.canvas = canvas;
		const context = canvas.getContext('webgpu') as unknown as GPUCanvasContext | null;
		if (!context) {
			throw new Error('OffscreenCanvas WebGPU context not available');
		}
		this.context = context;
		this.canvasFormat = format ?? navigator.gpu.getPreferredCanvasFormat();
		// Use negotiated device limit (not adapter), to avoid validation errors
		this.maxTextureDim = Math.max(1, this.device.limits.maxTextureDimension2D | 0);

		context.configure({
			device: this.device,
			format: this.canvasFormat,
			alphaMode: 'premultiplied'
		});

		this.recreateComposeTargets();
		this.ensurePipelines();
	}

	setCssLabelPadding(pixels: number) {
		this.cssLabelPaddingPx = Math.max(0, Math.floor(pixels || 0));
	}

	resize(width: number) {
		if (!this.canvas) return;
		// Clamp to device limits to prevent creating oversized textures
		const clampedW = Math.max(1, Math.min(width | 0, this.maxTextureDim));
		// We intentionally ignore incoming height and derive it from content during render.
		if (this.targetWidth === clampedW) return;
		this.targetWidth = clampedW;
		this.needsResize = true;
	}

	private recreateComposeTargets() {
		if (!this.canvas) return;
		this.composeTexture?.destroy();
		this.composeTexture = this.device.createTexture({
			size: { width: this.canvas.width, height: this.canvas.height },
			format: this.storageFormat,
			usage:
				GPUTextureUsage.RENDER_ATTACHMENT |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING
		});
		this.composeView = this.composeTexture.createView();
	}

	private ensurePipelines() {
		if (!this.blitPipeline) {
			this.blitBindGroupLayout = this.device.createBindGroupLayout({
				entries: [
					{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
					{ binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }
				]
			});
			const pipelineLayout = this.device.createPipelineLayout({
				bindGroupLayouts: [this.blitBindGroupLayout]
			});
			this.blitPipeline = this.device.createRenderPipeline({
				layout: pipelineLayout,
				vertex: {
					module: this.device.createShaderModule({ code: BLIT_WGSL }),
					entryPoint: 'vsMain'
				},
				fragment: {
					module: this.device.createShaderModule({ code: BLIT_WGSL }),
					entryPoint: 'fsMain',
					targets: [{ format: this.canvasFormat }]
				},
				primitive: { topology: 'triangle-list' }
			});
			this.blitSampler = this.device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
		}

		if (!this.computePipeline) {
			const layout = this.device.createBindGroupLayout({
				entries: [
					{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
					{
						binding: 1,
						visibility: GPUShaderStage.COMPUTE,
						storageTexture: { format: this.storageFormat, access: 'write-only' }
					},
					{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
					{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
				]
			});
			const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
			this.computePipeline = this.device.createComputePipeline({
				layout: pipelineLayout,
				compute: {
					module: this.device.createShaderModule({ code: COPY_COMPUTE_WGSL }),
					entryPoint: 'main'
				}
			});
		}

		// Reduction pipelines
		if (!this.reducePipelineStage1) {
			const layout1 = this.device.createBindGroupLayout({
				entries: [
					{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
					{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
					{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
				]
			});
			const pl1 = this.device.createPipelineLayout({ bindGroupLayouts: [layout1] });
			this.reducePipelineStage1 = this.device.createComputePipeline({
				layout: pl1,
				compute: {
					module: this.device.createShaderModule({ code: REDUCE_STAGE1_WGSL }),
					entryPoint: 'main'
				}
			});
		}

		if (!this.reducePipelineStage2) {
			const layout2 = this.device.createBindGroupLayout({
				entries: [
					{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
					{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
					{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
				]
			});
			const pl2 = this.device.createPipelineLayout({ bindGroupLayouts: [layout2] });
			this.reducePipelineStage2 = this.device.createComputePipeline({
				layout: pl2,
				compute: {
					module: this.device.createShaderModule({ code: REDUCE_STAGE2_WGSL }),
					entryPoint: 'main'
				}
			});
		}
	}

	private prepareMatches(matches: CaptureMatch[]): PreparedMatch[] {
		const prepared: PreparedMatch[] = [];
		for (const match of matches) {
			if (!match.tensor) continue;

			const dims = match.type === 'parameter' ? match.shape : match.shape.slice(1);

			let tileWidth = 1;
			let tileHeight = 1;
			let gridRows = 1;
			let gridCols = 1;

			if (dims.length === 0) {
				tileWidth = 1;
				tileHeight = 1;
				gridRows = 1;
				gridCols = 1;
			} else if (dims.length === 1) {
				tileWidth = dims[0];
				tileHeight = 1;
				gridRows = 1;
				gridCols = 1;
			} else if (dims.length === 2) {
				tileHeight = dims[0];
				tileWidth = dims[1];
				gridRows = 1;
				gridCols = 1;
			} else {
				// Lay out last two dims as 2D tile, third-from-last horizontally,
				// and any remaining leading dims stacked vertically.
				tileHeight = dims[dims.length - 2];
				tileWidth = dims[dims.length - 1];
				gridCols = dims[dims.length - 3];
				gridRows = dims.slice(0, Math.max(0, dims.length - 3)).reduce((a, b) => a * b, 1);
			}

			const width = gridCols * tileWidth;
			const height = gridRows * tileHeight;
			const scale = match.source.scale ?? 1.0;

			// Use precomputed buffer if available
			const buffer = match.buffer ?? match.tensor.gpuBuffer();
			match.tensor.__pistonDrop();
			if (!buffer) continue;

			prepared.push({
				match: { ...match, buffer } as CaptureMatch & { buffer: GPUBuffer },
				gridRows,
				gridCols,
				tileWidth,
				tileHeight,
				width,
				height,
				scale
			});
		}
		return prepared;
	}

	private layoutMatches(prepared: PreparedMatch[]): MatchBox[] {
		const boxes: MatchBox[] = [];
		if (prepared.length === 0) return boxes;

		const labelCss = this.cssLabelPaddingPx | 0;
		const PAD = PADDING_PX | 0;
		const HALF_PAD = Math.floor(PAD / 2);

		// Fix the container width to the current canvas width; height very large so we
		// never overflow horizontally. Actual drawing will be clipped to the canvas height.
		const canvasW = (this.canvas?.width ?? this.targetWidth ?? 1) | 0;
		const containerInnerW = Math.max(1, canvasW - (GLOBAL_PADDING_PX << 1));
		const containerH = 1 << 30; // arbitrarily tall

		// Build batch rectangles for addArray, inflating each rectangle by PADDING_PX/2 on all sides,
		// and including the label vertical space above the tile.
		const rects: PackedRect[] = [];
		const dimsById = new Map<number, Omit<MatchBox, 'x' | 'y' | 'matchId'>>();

		// Helper to count power-of-two factors for 2-adic rewrapping
		const v2 = (n: number): number => {
			let c = 0;
			let x = Math.max(1, n | 0);
			while ((x & 1) === 0) {
				x >>= 1;
				c++;
			}
			return c;
		};

		for (const p of prepared) {
			const baseTileWidth = p.tileWidth | 0;
			const baseTileHeight = p.tileHeight | 0;
			const baseRows = p.gridRows | 0;
			const baseCols = p.gridCols | 0;
			const scaledW0 = Math.ceil(p.width * p.scale);

			// Compute required halving power to fit within inner container width (including padding)
			const needFactor = Math.max(1, (scaledW0 + PAD) / containerInnerW);
			const nNeeded = Math.max(0, Math.ceil(Math.log2(needFactor)));

			// Capacity from 2-adic factors of tile width and grid columns
			const aMax = v2(baseTileWidth);
			const bMax = v2(baseCols);
			const nCap = aMax + bMax;
			const nUse = Math.min(nNeeded, nCap);
			const a = Math.min(nUse, aMax);
			const b = Math.min(nUse - a, bMax);

			const adjTileWidth = Math.max(1, baseTileWidth >> a);
			const adjTileHeight = Math.max(1, baseTileHeight << a);
			const adjCols = Math.max(1, baseCols >> b);
			const adjRows = Math.max(1, baseRows << b);

			const scaledWidth = Math.ceil(adjCols * adjTileWidth * p.scale);
			const scaledHeight = Math.ceil(adjRows * adjTileHeight * p.scale);

			const reqW = Math.max(1, scaledWidth + PAD);
			const reqH = Math.max(1, scaledHeight + labelCss + PAD);
			const qid = p.match.queryIndex;
			rects.push({
				x: 0,
				y: 0,
				width: reqW,
				height: reqH,
				// id for caller use; set to queryIndex per request
				id: qid,
				// preserve original match id for mapping back to dims later
				matchId: p.match.matchId
			});
			dimsById.set(p.match.matchId, {
				width: scaledWidth,
				height: scaledHeight,
				match: p.match,
				tileWidth: adjTileWidth,
				tileHeight: adjTileHeight,
				gridRows: adjRows,
				gridCols: adjCols
			});
		}

		// Use MaxRectsPacker to pack within fixed width and very tall height
		const packer = new MaxRectsPacker<PackedRect>(containerInnerW, containerH, 0, {
			smart: false,
			pot: true,
			square: false,
			allowRotation: false,
			tag: false,
			border: 0,
			logic: PACKING_LOGIC.MAX_EDGE
		});
		packer.addArray(rects);
		const posById = new Map<number, { x: number; y: number }>();
		for (const bin of packer.bins) {
			for (const rect of bin.rects) {
				const mid = rect.matchId;
				posById.set(mid, { x: rect.x | 0, y: rect.y | 0 });
			}
		}

		// Preserve original order to match prepared[i] with boxes[i]
		for (const p of prepared) {
			const dims = dimsById.get(p.match.matchId)!;
			const pos = posById.get(p.match.matchId);

			const x = GLOBAL_PADDING_PX + ((pos?.x ?? 0) | 0) + HALF_PAD;
			const y = GLOBAL_PADDING_PX + ((pos?.y ?? 0) | 0) + HALF_PAD + labelCss;

			boxes.push({
				matchId: p.match.matchId,
				match: p.match,
				x,
				y,
				width: dims.width,
				height: dims.height,
				tileWidth: dims.tileWidth,
				tileHeight: dims.tileHeight,
				gridRows: dims.gridRows,
				gridCols: dims.gridCols
			});
		}

		return boxes;
	}

	async renderCapture(matches: CaptureMatch[]): Promise<{
		boxes: MatchBox[];
		statsById: Record<number, { mean: number; variance: number; l2: number }>;
		width: number;
		height: number;
	}> {
		if (!this.context || !this.canvas || !this.composeTexture || !this.composeView)
			return { boxes: [], statsById: {}, width: 1, height: 1 };

		// Apply pending resize (width only) before computing layout
		if (this.needsResize && this.canvas) {
			this.canvas.width = this.targetWidth;
			// We delay height adjustment until after computing layout
			this.recreateComposeTargets();
			this.needsResize = false;
		}

		this.ensurePipelines();

		// Track transient per-frame buffers to explicitly destroy after GPU work completes
		const transientBuffers: GPUBuffer[] = [];

		const prepared = this.prepareMatches(matches);
		const boxes = this.layoutMatches(prepared);

		// Derive canvas height from content (max y + height), clamped to device limits
		let derivedHeight = 1;
		for (const b of boxes) {
			const bottom = (b.y | 0) + (b.height | 0);
			if (bottom > derivedHeight) derivedHeight = bottom;
		}
		derivedHeight = Math.max(
			1,
			// Add a small global padding at bottom
			Math.min((derivedHeight + (GLOBAL_PADDING_PX + PADDING_PX / 2)) | 0, this.maxTextureDim)
		);
		if ((this.canvas.height | 0) !== derivedHeight) {
			this.canvas.height = derivedHeight;
			// Recreate compose targets to match new height
			this.recreateComposeTargets();
		}

		const readbacks: { buffer: GPUBuffer; matchId: number }[] = [];

		const encoder = this.device.createCommandEncoder();
		let submissionSucceeded = false;

		try {
			// Clear compose target
			{
				const pass = encoder.beginRenderPass({
					colorAttachments: [
						{
							view: this.composeView,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: 'clear',
							storeOp: 'store'
						}
					]
				});
				pass.end();
			}

			// For each match: copy tensor slice to a readable storage buffer, run GPU reduction to
			// compute mean/variance, then run compute to paint into compose texture using the computed
			// stats
			for (let i = 0; i < prepared.length; i++) {
				const p = prepared[i];
				const box = boxes[i];

				const elementCount = Math.max(
					1,
					box.gridRows * box.gridCols * box.tileWidth * box.tileHeight
				);
				const byteSize = elementCount * 4;

				const readable = this.device.createBuffer({
					size: byteSize,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
				});
				transientBuffers.push(readable);

				// Copy the selected batch index for non-parameter tensors; parameters ignore batchIndex
				const isParameter = p.match.type === 'parameter';
				const batchCount = isParameter ? 1 : Math.max(1, p.match.shape[0] | 0);
				const requestedBatchIndex = isParameter ? 0 : (p.match.batchIndex ?? 0);
				const clampedBatchIndex = isParameter
					? 0
					: Math.max(0, Math.min(batchCount - 1, requestedBatchIndex | 0));
				const offsetBytes = isParameter ? 0 : (clampedBatchIndex * elementCount * 4) >>> 0;
				const maxBytes = Math.max(0, p.match.buffer.size - offsetBytes);
				const copyBytes = Math.min(byteSize, maxBytes);
				if (copyBytes > 0) {
					encoder.copyBufferToBuffer(p.match.buffer, offsetBytes, readable, 0, copyBytes);
				}

				// Reduction to compute mean/variance on GPU
				const WG_SIZE = 256;
				const ELEMENTS_PER_THREAD = 4;
				const perGroup = WG_SIZE * ELEMENTS_PER_THREAD;
				const numPartials = Math.max(1, Math.ceil(elementCount / perGroup));

				const partialsBuffer = this.device.createBuffer({
					size: numPartials * 8,
					usage: GPUBufferUsage.STORAGE
				});
				transientBuffers.push(partialsBuffer);

				const finalStats = this.device.createBuffer({
					size: 12,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
				});
				transientBuffers.push(finalStats);

				const reduceParams = new Uint32Array([
					elementCount >>> 0,
					numPartials >>> 0,
					WG_SIZE >>> 0,
					ELEMENTS_PER_THREAD >>> 0
				]);
				const reduceUniform = this.device.createBuffer({
					size: Math.ceil(reduceParams.byteLength / 16) * 16,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				});
				transientBuffers.push(reduceUniform);
				this.device.queue.writeBuffer(
					reduceUniform,
					0,
					reduceParams.buffer,
					reduceParams.byteOffset,
					reduceParams.byteLength
				);

				// Stage 1 reduction
				{
					const bg = this.device.createBindGroup({
						layout: this.reducePipelineStage1!.getBindGroupLayout(0),
						entries: [
							{ binding: 0, resource: { buffer: readable } },
							{ binding: 1, resource: { buffer: partialsBuffer } },
							{ binding: 2, resource: { buffer: reduceUniform } }
						]
					});
					const pass = encoder.beginComputePass();
					pass.setPipeline(this.reducePipelineStage1!);
					pass.setBindGroup(0, bg);
					pass.dispatchWorkgroups(numPartials);
					pass.end();
				}

				// Stage 2 reduction
				{
					const bg = this.device.createBindGroup({
						layout: this.reducePipelineStage2!.getBindGroupLayout(0),
						entries: [
							{ binding: 0, resource: { buffer: partialsBuffer } },
							{ binding: 1, resource: { buffer: finalStats } },
							{ binding: 2, resource: { buffer: reduceUniform } }
						]
					});
					const pass = encoder.beginComputePass();
					pass.setPipeline(this.reducePipelineStage2!);
					pass.setBindGroup(0, bg);
					pass.dispatchWorkgroups(1);
					pass.end();
				}

				// Copy stats to a CPU-readable buffer
				const readback = this.device.createBuffer({
					size: 12,
					usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
				});
				readbacks.push({ buffer: readback, matchId: p.match.matchId });
				encoder.copyBufferToBuffer(finalStats, 0, readback, 0, 12);

				// Paint compute using computed stats
				const scale = Math.fround(p.scale);
				const scaleU32 = new Uint32Array(new Float32Array([scale]).buffer)[0];
				const eps = Math.fround(1e-6);
				const epsU32 = new Uint32Array(new Float32Array([eps]).buffer)[0];
				const uniformData = new Uint32Array([
					box.x >>> 0,
					box.y >>> 0,
					box.tileWidth >>> 0,
					box.tileHeight >>> 0,
					box.gridRows >>> 0,
					box.gridCols >>> 0,
					scaleU32 >>> 0,
					epsU32 >>> 0
				]);
				const uniform = this.device.createBuffer({
					size: Math.ceil(uniformData.byteLength / 16) * 16,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				});
				transientBuffers.push(uniform);
				this.device.queue.writeBuffer(
					uniform,
					0,
					uniformData.buffer,
					uniformData.byteOffset,
					uniformData.byteLength
				);

				const bindGroup = this.device.createBindGroup({
					layout: this.computePipeline!.getBindGroupLayout(0),
					entries: [
						{ binding: 0, resource: { buffer: readable } },
						{ binding: 1, resource: this.composeView as GPUTextureView },
						{ binding: 2, resource: { buffer: uniform } },
						{ binding: 3, resource: { buffer: finalStats } }
					]
				});

				const pass = encoder.beginComputePass();
				pass.setPipeline(this.computePipeline!);
				pass.setBindGroup(0, bindGroup);
				const renderWidth = box.width | 0;
				const renderHeight = box.height | 0;
				// Clip dispatch region to current compose texture bounds to avoid OOB writes
				const targetW = this.canvas.width | 0;
				const targetH = this.canvas.height | 0;
				const clipW = Math.max(0, Math.min(renderWidth, Math.max(0, targetW - box.x)));
				const clipH = Math.max(0, Math.min(renderHeight, Math.max(0, targetH - box.y)));
				if (clipW <= 0 || clipH <= 0) {
					pass.end();
					continue;
				}
				const dispatchX = Math.ceil(clipW / 8);
				const dispatchY = Math.ceil(clipH / 8);
				pass.dispatchWorkgroups(dispatchX, dispatchY);
				pass.end();
			}

			// Blit compose texture to the current swapchain texture
			const currentTexture = this.context.getCurrentTexture();
			const currentView = currentTexture.createView();

			const blitBindGroup = this.device.createBindGroup({
				layout: this.blitBindGroupLayout!,
				entries: [
					{ binding: 0, resource: this.composeView as GPUTextureView },
					{ binding: 1, resource: this.blitSampler as GPUSampler }
				]
			});

			{
				const pass = encoder.beginRenderPass({
					colorAttachments: [
						{
							view: currentView,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: 'clear',
							storeOp: 'store'
						}
					]
				});
				pass.setPipeline(this.blitPipeline!);
				pass.setBindGroup(0, blitBindGroup);
				pass.draw(6, 1, 0, 0);
				pass.end();
			}

			this.device.queue.submit([encoder.finish()]);
			submissionSucceeded = true;
		} catch (_err) {
			// If we failed before submitting GPU work, destroy transient buffers immediately
			if (!submissionSucceeded) {
				for (const buf of transientBuffers) {
					try {
						buf.destroy();
					} catch (_e) {
						/* ignore */
					}
				}
			}
			throw _err;
		}

		// Wait for GPU work to complete before mapping readback buffers
		await this.device.queue.onSubmittedWorkDone();

		// Schedule explicit destruction of transient buffers once the GPU has
		// finished consuming this submission. This avoids leaking GPU memory
		// across frames if GC is delayed.
		void this.device.queue.onSubmittedWorkDone().then(() => {
			for (const buf of transientBuffers) {
				try {
					buf.destroy();
				} catch (_e) {
					// Ignore destruction errors; resource may already be collected
				}
			}
		});

		const statsById: Record<number, { mean: number; variance: number; l2: number }> = {};
		for (const rb of readbacks) {
			try {
				await rb.buffer.mapAsync(GPUMapMode.READ);
				const range = rb.buffer.getMappedRange();
				const f32 = new Float32Array(range.slice(0));
				statsById[rb.matchId] = { mean: f32[0] ?? 0, variance: f32[1] ?? 1, l2: f32[2] ?? 0 };
				rb.buffer.unmap();
				rb.buffer.destroy();
			} catch (_e) {
				// Ignore mapping errors for individual buffers
				try {
					rb.buffer.unmap();
				} catch (__ignore) {
					void __ignore;
				}
				try {
					rb.buffer.destroy();
				} catch (__ignore2) {
					void __ignore2;
				}
			}
		}

		return {
			boxes,
			statsById,
			width: this.canvas?.width ?? 1,
			height: this.canvas?.height ?? 1
		};
	}

	dispose() {
		// Destroy long-lived GPU resources and release references
		try {
			this.composeTexture?.destroy();
		} catch (_e) {
			void _e;
		}
		try {
			this.context?.unconfigure();
		} catch (_e) {
			void _e;
		}

		this.composeTexture = null;
		this.composeView = null;
		this.blitPipeline = null;
		this.computePipeline = null;
		this.blitBindGroupLayout = null;
		this.blitSampler = null;
		this.context = null;
		this.canvas = null;
	}
}

const BLIT_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
    vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
  );
  var out: VSOut;
  out.pos = vec4f(positions[vid], 0.0, 1.0);
  out.uv = uvs[vid];
  return out;
}

@group(0) @binding(0) var img : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  return textureSample(img, samp, in.uv);
}
`;

// Stage 1: parallel partials of sum and sum of squares
const REDUCE_STAGE1_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> partials : array<vec2<f32>>;

struct Params {
  n : u32,
  numPartials : u32,
  wgSize : u32,
  elemsPerThread : u32,
};

@group(0) @binding(2) var<uniform> P : Params;

var<workgroup> ssum : array<f32, 256>;
var<workgroup> ssum2 : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let groupIndex = wid.x;
  let localIndex = lid.x;
  let start = groupIndex * (P.wgSize * P.elemsPerThread) + localIndex;

  var sum : f32 = 0.0;
  var sum2 : f32 = 0.0;
  for (var i: u32 = 0u; i < P.elemsPerThread; i = i + 1u) {
    let idx = start + i * P.wgSize;
    if (idx < P.n) {
      let v = src[idx];
      sum = sum + v;
      sum2 = sum2 + v * v;
    }
  }

  ssum[localIndex] = sum;
  ssum2[localIndex] = sum2;
  workgroupBarrier();

  var offset : u32 = 256u / 2u;
  loop {
    if (offset == 0u) { break; }
    if (localIndex < offset) {
      ssum[localIndex] = ssum[localIndex] + ssum[localIndex + offset];
      ssum2[localIndex] = ssum2[localIndex] + ssum2[localIndex + offset];
    }
    workgroupBarrier();
    offset = offset / 2u;
  }

  if (localIndex == 0u) {
    partials[groupIndex] = vec2<f32>(ssum[0], ssum2[0]);
  }
}
`;

// Stage 2: finalize mean and variance from partials
const REDUCE_STAGE2_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> partials : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outStats : array<f32>;

struct Params {
  n : u32,
  numPartials : u32,
  wgSize : u32,
  elemsPerThread : u32,
};

@group(0) @binding(2) var<uniform> P : Params;

var<workgroup> ssum : array<f32, 256>;
var<workgroup> ssum2 : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let localIndex = lid.x;
  var sum : f32 = 0.0;
  var sum2 : f32 = 0.0;
  for (var i: u32 = localIndex; i < P.numPartials; i = i + 256u) {
    let p = partials[i];
    sum = sum + p.x;
    sum2 = sum2 + p.y;
  }
  ssum[localIndex] = sum;
  ssum2[localIndex] = sum2;
  workgroupBarrier();

  var offset : u32 = 256u / 2u;
  loop {
    if (offset == 0u) { break; }
    if (localIndex < offset) {
      ssum[localIndex] = ssum[localIndex] + ssum[localIndex + offset];
      ssum2[localIndex] = ssum2[localIndex] + ssum2[localIndex + offset];
    }
    workgroupBarrier();
    offset = offset / 2u;
  }

  if (localIndex == 0u) {
    let n = max(1.0, f32(P.n));
    let mean = ssum[0] / n;
    let ex2 = ssum2[0] / n;
    let variance = max(0.0, ex2 - mean * mean);
    let l2 = sqrt(ssum2[0]);
    outStats[0] = mean;
    outStats[1] = variance;
    outStats[2] = l2;
  }
}
`;

const COPY_COMPUTE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var outImg : texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
  originX : u32,
  originY : u32,
  tileW : u32,
  tileH : u32,
  gridRows : u32,
  gridCols : u32,
  scaleBits : u32,
  epsBits : u32,
};

@group(0) @binding(2) var<uniform> U : Uniforms;
@group(0) @binding(3) var<storage, read> stats : array<f32>;

fn f32_from_bits(u: u32) -> f32 {
  return bitcast<f32>(u);
}

fn gamma_correct(v: vec3<f32>) -> vec3<f32> {
	let sign = select(vec3(-1.0), vec3(1.0), v >= vec3(0.0));
	let abs_v = abs(v);

	let power_term = sign * (1.055 * pow(abs_v, vec3(1.0/2.4)) - 0.055);
	let linear_term = 12.92 * v;

	return select(linear_term, power_term, abs_v > vec3(0.0031308));
}

fn oklch_to_srgb(oklch: vec3<f32>) -> vec3<f32> {
	// Convert OKLCH to OKLab
	let L = oklch.x;
	let C = oklch.y;
	let H = radians(oklch.z);

	let oklab = vec3<f32>(
			L,
			C * cos(H),
			C * sin(H)
	);

	// Combined matrices from Python calculation
	let oklab_to_lms = mat3x3<f32>(
			1.0000000000000000,   1.0000000000000000,   1.0000000000000000,
			0.3963377773761749,  -0.1055613458156586,  -0.0894841775298119,
			0.2158037573099136,  -0.0638541728258133,  -1.2914855480194092
	);

	// Combined xyz_to_rgb * lms_to_xyz matrix
	let xyz_rgb_lms = mat3x3<f32>(
			4.0767416360759583,  -1.2684379732850317,  -0.0041960761386756,
			-3.3077115392580616,   2.6097573492876887,  -0.7034186179359363,
			0.2309699031821043,  -0.3413193760026573,   1.7076146940746117
	);

	// Convert OKLab to LMS
	let lms = oklab_to_lms * oklab;

	// Apply cubic transform
	let lms_cubic = pow(lms, vec3(3.));

	// Convert directly to RGB using combined matrix
	let rgb_linear = xyz_rgb_lms * lms_cubic;

	// Apply gamma correction
	return gamma_correct(rgb_linear);
}

fn get_color(x: f32) -> vec3<f32> {
	let color1_pos = vec3f(0.0, 0.05, 20);
	let color2_pos = vec3f(0.62, 0.18, 41);
	let color1_neg = vec3f(0, 0.05, 247);
	let color2_neg = vec3f(0.62, 0.18, 267);

	// x is now in terms of standard deviations
	if (x < 0.0) {
		return mix(color1_neg, color2_neg, -x / 2.8);
	} else {
		return mix(color1_pos, color2_pos, x / 2.8);
	}
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let scale = max(0.0001, f32_from_bits(U.scaleBits));
  let eps = f32_from_bits(U.epsBits);
  let mean = stats[0u];
  let variance = stats[1u];
  let l2 = stats[2u];

  // Local pixel within this tile's box
  let lx = i32(gid.x);
  let ly = i32(gid.y);

  // Map local pixel to logical tensor pixel using scale
  let logicalX = i32(floor(f32(lx) / scale));
  let logicalY = i32(floor(f32(ly) / scale));

  let totalW = i32(U.tileW * U.gridCols);
  let totalH = i32(U.tileH * U.gridRows);

  if (logicalX >= totalW || logicalY >= totalH) {
    return;
  }

  let tileCol = logicalX / i32(U.tileW);
  let tileRow = logicalY / i32(U.tileH);
  let pxX = logicalX % i32(U.tileW);
  let pxY = logicalY % i32(U.tileH);

  let leadingIndex = tileRow * i32(U.gridCols) + tileCol;
  let index = leadingIndex * i32(U.tileW) * i32(U.tileH) + pxY * i32(U.tileW) + pxX;

  let v = src[index];

  // z-score color mapping computed on-GPU: invStd = rsqrt(var + eps)
  let invStd = inverseSqrt(max(0.0, variance + eps));
  let z = (v - mean) * invStd;
  // let z = v * invStd;
  // let zc = clamp(z, -3.0, 3.0) / 3.0;
  // let blue = max(0.0, zc);
  // let red = max(0.0, -zc);
  // let color = vec4<f32>(red, 0.0, blue, 1.0);
	let color = vec4<f32>(oklch_to_srgb(get_color(z)), 1.0);

  // Write to destination at tile origin offset
  let outX = U.originX + gid.x;
  let outY = U.originY + gid.y;
  textureStore(outImg, vec2u(outX, outY), color);
}
`;
