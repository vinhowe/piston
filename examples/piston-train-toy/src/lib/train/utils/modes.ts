import {
	PistonFunctionMode,
	PistonMarkStepMode,
	WeakMarkStepMode,
	WeakTensorFunctionMode,
	type WeakTensorFunctionModeOptions
} from '@piston-ml/piston-web';
import * as piston from '@piston-ml/piston-web';

export class DebugMode extends PistonFunctionMode {
	constructor(public debugEnabled: boolean) {
		super();
	}

	_pistonFunction<T>(
		func: <FT>(...args: unknown[]) => FT | Promise<FT>,
		_types: unknown[],
		args: unknown[],
		kwargs: Record<string, unknown>
	): T | Promise<T> {
		if (this.debugEnabled) {
			console.log(
				func.name,
				args.reduce((acc: number[], a) => {
					if (a instanceof piston.wasm.Tensor_wasm) {
						return [...acc, a.id];
					}
					return acc;
				}, [])
			);
		}

		const after = (result: T) => {
			if (result instanceof piston.wasm.Tensor_wasm) {
				console.log(func.name, 'result', result.id);
			}
			return result;
		};

		const result = func(...args, kwargs) as T | Promise<T>;
		if (result instanceof Promise) {
			return result.then(after) as Promise<T>;
		}

		return after(result) as T;
	}
}

export class WeakModeIfEnabled {
	private mode: WeakTensorFunctionMode | null = null;

	constructor(
		public enabled: boolean,
		public options: WeakTensorFunctionModeOptions
	) {
		if (enabled) {
			this.mode = new WeakTensorFunctionMode(options);
		}
	}

	markWeak<T>(input: T) {
		if (this.mode) {
			this.mode.markWeak(input);
		}
		return input;
	}

	pin<T>(input: T) {
		if (this.mode) {
			this.mode.pin(input);
		}
		return input;
	}

	[Symbol.dispose]() {
		if (this.mode) {
			this.mode[Symbol.dispose]();
		}
	}
}

export class MarkStepModeIfEnabled {
	private mode: PistonMarkStepMode | null = null;

	constructor(public enabled: boolean) {
		if (enabled) {
			this.mode = new WeakMarkStepMode();
		}
	}

	[Symbol.dispose]() {
		if (this.mode) {
			this.mode[Symbol.dispose]();
		}
	}
}

export class UniqueOperationsMode extends PistonFunctionMode {
	private uniqueOperations: Set<string> = new Set();

	constructor() {
		super();
	}

	_pistonFunction<T>(
		func: <FT>(...args: unknown[]) => FT | Promise<FT>,
		_types: unknown[],
		args: unknown[],
		kwargs: Record<string, unknown>
	) {
		this.uniqueOperations.add(func.name);
		return func(...args, kwargs) as T;
	}

	[Symbol.dispose]() {
		console.log('Unique operations:', this.uniqueOperations.size);
		console.log(JSON.stringify(Array.from(this.uniqueOperations), null, 2));
	}
}
