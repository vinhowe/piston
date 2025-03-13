use anyhow::Result;
use num_traits::AsPrimitive;
use piston::{shape, Device, HashMap, Shape, Tensor, TensorDType};

#[derive(Clone, Debug)]
pub struct KVEntry {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub entries: usize,
}

impl KVEntry {
    pub fn allocate<T: TensorDType + AsPrimitive<f32>>(
        shape: &Shape,
        device: &Device,
    ) -> Result<Self> {
        Ok(KVEntry {
            k_cache: Tensor::zeros::<T>(shape, device)?,
            v_cache: Tensor::zeros::<T>(shape, device)?,
            entries: 0,
        })
    }
}

#[derive(Clone, Debug)]
pub struct KVCache {
    entries: Vec<KVEntry>,
    use_kv_cache: bool,
    masks: HashMap<usize, Tensor>,
    device: Device,
    n_layers: usize,
    allocated: bool,
    shape: Shape,
}

impl std::ops::Index<usize> for KVCache {
    type Output = KVEntry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl KVCache {
    pub fn new<T: TensorDType + AsPrimitive<f32>>(
        n_layers: i32,
        use_kv_cache: bool,
        shape: Shape,
        device: &Device,
    ) -> Result<Self> {
        let mut entries = Vec::with_capacity(n_layers as _);
        // TODO: This is really bad; look at actual patterns for how people do KV caches
        let mut allocated = false;
        if use_kv_cache {
            for _ in 0..n_layers {
                entries.push(KVEntry::allocate::<T>(&shape, device)?);
            }
            allocated = true;
        }
        Ok(KVCache {
            entries,
            masks: HashMap::default(),
            device: device.clone(),
            n_layers: n_layers as _,
            use_kv_cache,
            allocated,
            shape,
        })
    }

    pub fn update(&mut self, offset: usize) {
        for entry in &mut self.entries {
            entry.entries += offset;
        }
    }

    pub fn entries(&self, layer: usize) -> usize {
        self.entries[layer].entries
    }

    pub fn reset(&mut self) {
        for entry in &mut self.entries {
            entry.entries = 0;
        }
    }

    pub fn use_kv_cache(&self) -> bool {
        self.use_kv_cache
    }

    pub fn set_use_kv_cache(&mut self, use_kv_cache: bool) -> Result<()> {
        self.use_kv_cache = use_kv_cache;
        if !use_kv_cache && self.allocated {
            self.entries.clear();
            self.allocated = false;
        } else if use_kv_cache && !self.allocated {
            for _ in 0..self.n_layers {
                self.entries
                    .push(KVEntry::allocate::<f32>(&self.shape, &self.device)?);
            }
            self.allocated = true;
        }
        Ok(())
    }

    pub fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            log::debug!("Using existing mask for {:?}", t);
            Ok(mask.clone())
        } else {
            log::debug!("Creating mask for {:?}", t);
            let ones = Tensor::ones::<f32>(&shape![t, t], &self.device)?;
            let mask = ones.tril(None)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}
