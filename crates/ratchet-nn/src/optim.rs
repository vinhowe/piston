use half::{bf16, f16};
use maybe_async::maybe_async;
use ratchet::{DType, Device, GradStore, ScopePusher, Var};

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<(Option<String>, Var)>, config: Self::Config) -> anyhow::Result<Self>;

    async fn step(&mut self, grads: &ratchet::GradStore, device: &Device) -> anyhow::Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> anyhow::Result<Self> {
        Self::new(vec![], config)
    }

    async fn backward_step(
        &mut self,
        grads: &mut GradStore,
        device: &Device,
    ) -> anyhow::Result<()> {
        self.step(grads, device).await
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> anyhow::Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| (None, v.clone())).collect();
        Self::new(vars, config)
    }
}

#[derive(Debug)]
pub struct SGD {
    vars: Vec<(Option<String>, Var)>,
    learning_rate: f64,
}

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<(Option<String>, Var)>, learning_rate: f64) -> anyhow::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|(_, v)| v.as_tensor().dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    async fn step(&mut self, grads: &ratchet::GradStore, device: &Device) -> anyhow::Result<()> {
        let mut updates = Vec::new();
        {
            let _scope_guard = ScopePusher::new("optim:SGD");
            for (_, var) in &self.vars {
                let _scope_guard = optim_var_scope_guard(var);
                if let Some(grad) = grads.get(var.as_tensor()) {
                    let update =
                        (var.as_tensor().clone() - (grad.clone() * self.learning_rate as f32)?)?;
                    updates.push(var.set(update));
                }
            }
        }

        if let Ok(gpu_device) = device.try_gpu() {
            gpu_device.mark_step().await;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
    label: Option<String>,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<(Option<String>, Var)>, params: ParamsAdamW) -> anyhow::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|(_, var)| var.as_tensor().dtype().is_float())
            .map(|(label, var)| {
                let var_t = var.as_tensor();
                let dtype = var_t.dtype();
                let shape = var_t.shape();
                let device = var_t.device();
                let (first_moment, second_moment) = match dtype {
                    DType::F32 => (
                        Var::zeros::<f32>(shape, device)?,
                        Var::zeros::<f32>(shape, device)?,
                    ),
                    DType::F16 => (
                        Var::zeros::<f16>(shape, device)?,
                        Var::zeros::<f16>(shape, device)?,
                    ),
                    DType::BF16 => (
                        Var::zeros::<bf16>(shape, device)?,
                        Var::zeros::<bf16>(shape, device)?,
                    ),
                    _ => return Err(anyhow::anyhow!("Unsupported dtype for AdamW: {:?}", dtype)),
                };
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                    label,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }
    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    async fn step(&mut self, grads: &ratchet::GradStore, device: &Device) -> anyhow::Result<()> {
        // This makes sure we keep references to the copy tensors.
        let mut updates = Vec::new();
        {
            let _scope_guard = ScopePusher::new("optim:AdamW");
            self.step_t += 1;
            let lr = self.params.lr;
            let lambda = self.params.weight_decay;
            let lr_lambda = lr * lambda;
            let beta1 = self.params.beta1;
            let beta2 = self.params.beta2;
            let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
            let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));

            for var in self.vars.iter_mut() {
                let _scope_guard = optim_var_scope_guard(&var.var);
                let theta = &var.var;
                let m = &var.first_moment;
                let v = &var.second_moment;

                // println!("Optimizer stepping: {:?}", var.label);
                // println!("Theta op: {:?}", theta.as_tensor().op());
                // println!("Theta id: {:?}", theta.as_tensor().id());

                if let Some(g) = grads.get(theta.as_tensor()) {
                    let next_m = ((m.as_tensor().clone() * beta1 as f32)?
                        + (g.clone() * (1.0 - beta1 as f32))?)?;
                    let next_v = ((v.as_tensor().clone() * beta2 as f32)?
                        + (g.clone().square()? * (1.0 - beta2 as f32))?)?;
                    let m_hat = (next_m.clone() * scale_m as f32)?;
                    let v_hat = (next_v.clone() * scale_v as f32)?;
                    let next_theta = (theta.as_tensor().clone() * (1f32 - lr_lambda as f32))?;
                    let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps as f32)?)?;
                    let next_theta = (next_theta - (adjusted_grad.clone() * lr as f32)?)?;

                    // This ensures we keep references to the copy tensors.
                    updates.push((theta.set(next_theta), m.set(next_m), v.set(next_v)));
                }
            }
        }

        // Finalize all the tensors we just built above.
        if let Ok(gpu) = device.try_gpu() {
            gpu.mark_step().await?;
        }

        Ok(())
    }
}

impl AdamW {
    pub fn new_lr(vars: Vec<(Option<String>, Var)>, learning_rate: f64) -> anyhow::Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}

fn optim_var_scope_guard(var: &Var) -> ScopePusher {
    ScopePusher::new(
        format!(
            "for:({})",
            var.as_tensor()
                .scope()
                .as_ref()
                .unwrap_or(&"unknown".to_string())
        )
        .as_str(),
    )
}
