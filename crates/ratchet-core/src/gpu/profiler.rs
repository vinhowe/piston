use crate::{HashMap, TensorId};
#[cfg(feature = "gpu-profiling")]
use itertools::Itertools;
use maybe_async::maybe_async;
#[cfg(feature = "gpu-profiling")]
use tabled::settings::{object::Rows, Alignment, Modify, Panel, Style};
#[cfg(feature = "gpu-profiling")]
use tabled::{Table, Tabled};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::QuerySet;

use super::WgpuDevice;

#[cfg(feature = "gpu-profiling")]
//used for formatting table cells
fn float2(n: &f64) -> String {
    format!("{:.2}", n)
}

#[cfg(feature = "gpu-profiling")]
#[derive(Tabled)]
struct SummaryTableEntry {
    #[tabled(rename = "Op Type")]
    op_type: String,
    #[tabled(rename = "Elapsed Time (ns)")]
    elapsed: usize,
    #[tabled(rename = "Count")]
    count: usize,
    #[tabled(rename = "Avg. Time (ns)")]
    avg_elapsed: usize,
    #[tabled(rename = "% of Runtime", display_with = "float2")]
    percent_runtime: f64,
}

#[cfg(feature = "gpu-profiling")]
pub fn build_summary_table(
    elapsed_map: HashMap<String, usize>,
    op_counts: HashMap<String, usize>,
) -> Table {
    let total_elapsed: usize = elapsed_map.values().sum();

    let mut elapsed: Vec<SummaryTableEntry> = elapsed_map
        .into_iter()
        .map(|(op_type, elapsed)| SummaryTableEntry {
            op_type: op_type.clone(),
            elapsed,
            count: *op_counts.get(&op_type).unwrap(),
            avg_elapsed: elapsed / op_counts.get(&op_type).unwrap(),
            percent_runtime: elapsed as f64 / total_elapsed as f64 * 100.0,
        })
        .collect();

    elapsed.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));

    let total = elapsed.iter().map(|e| e.elapsed).sum::<usize>() / 1_000;

    Table::new(&elapsed)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
        .with(Panel::footer(format!("{} total runtime (μs)", total)))
        .to_owned()
}

#[cfg(feature = "gpu-profiling")]
#[derive(Tabled)]
struct IndividualTableEntry {
    #[tabled(rename = "Node ID")]
    node_id: usize,
    #[tabled(rename = "Op Type")]
    op_type: String,
    #[tabled(rename = "Elapsed Time (ns)")]
    elapsed: usize,
    #[tabled(rename = "% of Runtime", display_with = "float2")]
    percent_runtime: f64,
}

#[cfg(feature = "gpu-profiling")]
pub fn build_individual_table(elapsed_map: HashMap<usize, (String, usize)>) -> Table {
    let total_elapsed: usize = elapsed_map.values().map(|(_, e)| e).sum();

    let mut elapsed: Vec<IndividualTableEntry> = elapsed_map
        .into_iter()
        .map(|(node_id, (op_type, elapsed))| IndividualTableEntry {
            node_id,
            op_type,
            elapsed,
            percent_runtime: elapsed as f64 / total_elapsed as f64 * 100.0,
        })
        .collect();

    elapsed.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));

    let total = elapsed.iter().map(|e| e.elapsed).sum::<usize>() / 1_000;

    Table::new(&elapsed)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
        .with(Panel::footer(format!("{} total runtime (μs)", total)))
        .to_owned()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter_with_clone))]
#[derive(Debug, Clone)]
pub struct ExportedTensorProfilingEntry {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub id: TensorId,
    pub kernel_name: String,
    pub elapsed: usize,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ExportedTensorProfilingEntry {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> TensorId {
        self.id
    }
}

pub struct Profiler {
    device: WgpuDevice,
    query_set: QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    query_index: u32,
    timestamp_period: f32,
    #[cfg(feature = "gpu-profiling")]
    query_to_node: HashMap<(u32, u32), (usize, String)>,
    #[cfg(not(feature = "gpu-profiling"))]
    query_to_tensor: HashMap<(u32, u32), (TensorId, String)>,
}

impl Profiler {
    pub fn new(device: WgpuDevice, count: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            count: count * 2,
            ty: wgpu::QueryType::Timestamp,
            label: Some("PerfTimestamps"),
        });
        let timestamp_period = device.queue().get_timestamp_period();

        let buffer_size = (count as usize * 2 * std::mem::size_of::<u64>()) as u64;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfTimestamps"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let destination_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfTimestamps"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            device,
            query_set,
            resolve_buffer,
            destination_buffer,
            query_index: 0,
            timestamp_period,
            #[cfg(feature = "gpu-profiling")]
            query_to_node: HashMap::with_capacity_and_hasher(count as usize, Default::default()),
            #[cfg(not(feature = "gpu-profiling"))]
            query_to_tensor: HashMap::with_capacity_and_hasher(count as usize, Default::default()),
        }
    }

    pub fn create_timestamp_queries(
        &mut self,
        #[cfg(feature = "gpu-profiling")] id: usize,
        #[cfg(not(feature = "gpu-profiling"))] id: TensorId,
        name: &str,
    ) -> wgpu::ComputePassTimestampWrites {
        let beginning_index = self.query_index;
        self.query_index += 1;
        let end_index = self.query_index;
        self.query_index += 1;

        let timestamp_writes = wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(beginning_index),
            end_of_pass_write_index: Some(end_index),
        };

        #[cfg(feature = "gpu-profiling")]
        self.query_to_node
            .insert((beginning_index, end_index), (id, name.to_string()));
        #[cfg(not(feature = "gpu-profiling"))]
        self.query_to_tensor
            .insert((beginning_index, end_index), (id, name.to_string()));

        timestamp_writes
    }

    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..self.query_index,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    #[cfg(feature = "gpu-profiling")]
    fn summary_table(&self, timestamps: &[u64]) {
        let mut elapsed_map = HashMap::new();
        let mut op_counts = HashMap::new();
        for (idx, (begin, end)) in timestamps.iter().tuples().enumerate() {
            let elapsed_ns = (end - begin) as f64 * self.timestamp_period as f64;
            let (_id, op_type) = self
                .query_to_node
                .get(&(idx as u32 * 2, idx as u32 * 2 + 1))
                .unwrap();
            elapsed_map
                .entry(op_type.to_string())
                .and_modify(|e| *e += elapsed_ns as usize)
                .or_insert(elapsed_ns as usize);
            op_counts
                .entry(op_type.to_string())
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }

        println!("{}", build_summary_table(elapsed_map, op_counts));
    }

    #[cfg(feature = "gpu-profiling")]
    fn node_table(&self, timestamps: &[u64]) {
        let mut node_map = HashMap::new();
        for (idx, (begin, end)) in timestamps.iter().tuples().enumerate() {
            let elapsed_ns = (end - begin) as f64 * self.timestamp_period as f64;
            let (id, op_type) = self
                .query_to_node
                .get(&(idx as u32 * 2, idx as u32 * 2 + 1))
                .unwrap();
            node_map
                .entry(*id)
                .and_modify(|(_, e)| *e += elapsed_ns as usize)
                .or_insert((op_type.to_string(), elapsed_ns as usize));
        }

        println!("{}", build_individual_table(node_map));
    }

    #[cfg(feature = "gpu-profiling")]
    pub fn print_timestamps(&self, summary: bool) {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::Maintain::Wait);
        let timestamp_view = self
            .destination_buffer
            .slice(
                ..(std::mem::size_of::<u64>() * self.query_index as usize) as wgpu::BufferAddress,
            )
            .get_mapped_range();

        let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);

        if summary {
            self.summary_table(timestamps);
        } else {
            self.node_table(timestamps);
        }
    }

    #[cfg(not(feature = "gpu-profiling"))]
    #[maybe_async]
    pub async fn read_timestamps(&self) -> HashMap<TensorId, ExportedTensorProfilingEntry> {
        // I guess in principle we could do all of this in a shader...
        #[cfg(target_arch = "wasm32")]
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        #[cfg(not(target_arch = "wasm32"))]
        let (tx, rx) = std::sync::mpsc::channel();
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |_| {
                tx.send(()).expect("Failed to sync for profiling");
            });
        self.device.poll(wgpu::Maintain::Wait);
        #[cfg(target_arch = "wasm32")]
        rx.receive().await.unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        rx.recv().unwrap();
        let timestamp_view = self
            .destination_buffer
            .slice(
                ..(std::mem::size_of::<u64>() * self.query_index as usize) as wgpu::BufferAddress,
            )
            .get_mapped_range();

        let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);
        let mut exported_tensors = HashMap::default();
        for (idx, chunk) in timestamps.chunks(2).enumerate() {
            if let &[begin, end] = chunk {
                let elapsed_ns = (end - begin) as f64 * self.timestamp_period as f64;
                let (id, name) = self
                    .query_to_tensor
                    .get(&(idx as u32 * 2, idx as u32 * 2 + 1))
                    .unwrap();
                exported_tensors.insert(
                    *id,
                    ExportedTensorProfilingEntry {
                        id: *id,
                        kernel_name: name.clone(),
                        elapsed: elapsed_ns as usize,
                    },
                );
            }
        }
        exported_tensors
    }
}
