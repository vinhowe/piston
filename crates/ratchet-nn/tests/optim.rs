use ratchet::{
    shape,
    test_utils::{to_vec0_round, to_vec1_round},
    Device, DeviceRequest, Tensor, Var,
};
use ratchet_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW};

thread_local! {
    static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
}

#[test]
fn adamw_linear_regression() -> anyhow::Result<()> {
    let device = GPU_DEVICE.with(|d| d.clone());
    let w_gen = Tensor::from_data(vec![3f32, 1.], shape![1, 2], Device::CPU).to(&device)?;
    let b_gen = Tensor::from_data(vec![-2f32], shape![1, 1], Device::CPU).to(&device)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::from_data(
        vec![2f32, 1., 7., 4., -4., 12., 5., 8.],
        shape![4, 2],
        Device::CPU,
    );
    let sample_xs = sample_xs.to(&device)?;
    let sample_ys = gen.schedule(sample_xs.clone())?;

    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::from_tensor(
        &Tensor::from_data(vec![0f32, 0.], shape![1, 2], Device::CPU).to(&device)?,
    )?;
    // let b = Var::from_data(vec![0f32], shape![1], Device::CPU);
    let b =
        Var::from_tensor(&Tensor::from_data(vec![0f32], shape![1, 1], Device::CPU).to(&device)?)?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(
        vec![
            (Some(String::from("b")), b.clone()),
            (Some(String::from("w")), w.clone()),
        ],
        params,
    )?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.schedule(sample_xs.clone())?;
        let loss = ys.sub(sample_ys.clone())?.square()?.sum(&[0])?;
        // ratchet::plot::render_to_file(&loss, "forward-pre-schedule.svg").unwrap();
        let mut grads = loss.backward()?;
        // ratchet::plot::render_to_file(&loss, "forward-post-schedule.svg").unwrap();
        let loss_cpu = loss.clone().resolve()?.to(&Device::CPU)?;
        let loss_vec = loss_cpu.to_vec::<f32>()?;
        println!("loss: {:?}", loss_vec[0]);
        let b = b.as_tensor().to(&Device::CPU)?;
        let w = w.as_tensor().to(&Device::CPU)?;
        println!("b: {:?}, w: {:?}", b.to_vec::<f32>(), w.to_vec::<f32>());
        opt.backward_step(&mut grads, device.try_gpu()?)?;
    }
    let b = b.as_tensor().to(&Device::CPU)?;
    let w = w.as_tensor().to(&Device::CPU)?;
    println!("b: {:?}, w: {:?}", b.to_vec::<f32>(), w.to_vec::<f32>());
    assert_eq!(to_vec0_round(&b, 4)?, 0.7872);
    assert_eq!(to_vec1_round(&w, 4)?, &[2.7257, 0.7097]);
    Ok(())
}

#[test]
fn test_intermediate() -> anyhow::Result<()> {
    let device = GPU_DEVICE.with(|d| d.clone());
    let w_gen = Tensor::from_data(vec![3f32, 1.], shape![1, 2], Device::CPU).to(&device)?;
    let b_gen = Tensor::from_data(vec![-2f32], shape![1, 1], Device::CPU).to(&device)?;
    let gen = Linear::new(w_gen.clone(), Some(b_gen.clone()));
    let sample_xs = Tensor::from_data(
        vec![2f32, 1., 7., 4., -4., 12., 5., 8.],
        shape![4, 2],
        Device::CPU,
    );
    let sample_xs = sample_xs.to(&device)?;
    let sample_ys = gen.schedule(sample_xs.clone())?.resolve()?;

    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::from_tensor(
        &Tensor::from_data(vec![0f32, 0.], shape![1, 2], Device::CPU).to(&device)?,
    )?;
    // let b = Var::from_data(vec![0f32], shape![1], Device::CPU);
    let b =
        Var::from_tensor(&Tensor::from_data(vec![0f32], shape![1, 1], Device::CPU).to(&device)?)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));

    let ys = lin.schedule(sample_xs.clone())?;
    let loss = ys
        .sub(sample_ys.clone())?
        .square()?
        // .sum_last_dim(0)?
        .resolve()?;

    // Print loss
    println!("loss: {:?}", loss.to(&Device::CPU)?.to_vec::<f32>()?);

    let gen_grads = loss.backward()?;
    for (i, (_id, g)) in gen_grads.iter().enumerate() {
        let g_clone = g.clone().resolve()?;
        println!(
            "gen_grads[{}]: (op {:?}) {:?}",
            i,
            g_clone.clone().op().name(),
            g_clone.resolve()?.to(&Device::CPU)?.to_vec::<f32>()?
        );
    }
    Ok(())
}
