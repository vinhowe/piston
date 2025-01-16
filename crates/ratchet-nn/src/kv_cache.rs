use ratchet::{shape, Device, HashMap, Shape, Tensor, TensorDType};

#[derive(Clone, Debug)]
pub struct KVEntry {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub entries: usize,
}

impl KVEntry {
    pub fn allocate<T: TensorDType>(shape: &Shape, device: &Device) -> Self {
        KVEntry {
            k_cache: Tensor::zeros::<T>(shape, device),
            v_cache: Tensor::zeros::<T>(shape, device),
            entries: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KVCache {
    entries: Vec<KVEntry>,
    use_kv_cache: bool,
    masks: HashMap<usize, Tensor>,
    device: Device,
}

impl std::ops::Index<usize> for KVCache {
    type Output = KVEntry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl KVCache {
    pub fn new<T: TensorDType>(
        n_layers: i32,
        use_kv_cache: bool,
        shape: Shape,
        device: &Device,
    ) -> Self {
        let mut entries = Vec::with_capacity(n_layers as _);
        for _ in 0..n_layers {
            entries.push(KVEntry::allocate::<T>(&shape, device));
        }
        KVCache {
            entries,
            masks: HashMap::default(),
            device: device.clone(),
            use_kv_cache,
        }
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

    pub fn mask(&mut self, t: usize) -> anyhow::Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            log::debug!("Using existing mask for {:?}", t);
            Ok(mask.clone())
        } else {
            log::debug!("Creating mask for {:?}", t);
            log::debug!("masks: {:?}", self.masks);
            let ones = Tensor::ones::<f32>(&shape![t, t], &self.device);
            let mask = ones.triu(Some(1))?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}
