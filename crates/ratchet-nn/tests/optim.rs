use ratchet::{
    shape,
    test_utils::{to_vec0_round, to_vec1_round},
    Device, DeviceRequest, Tensor, Var,
};
use ratchet_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, SGD};

type OptimizerFactory<O> = fn(Vec<(Option<String>, Var)>) -> anyhow::Result<O>;

fn run_linear_regression<O: Optimizer>(optimizer: OptimizerFactory<O>) -> anyhow::Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    let device = Device::request_device(DeviceRequest::GPU).unwrap();
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
    let w = Var::zeros::<f32>(&shape![1, 2], &device);
    let b = Var::zeros::<f32>(&shape![1, 1], &device);
    let mut opt = optimizer(vec![
        (Some(String::from("b")), b.clone()),
        (Some(String::from("w")), w.clone()),
    ])?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));

    for _step in 0..100 {
        let ys = lin.schedule(sample_xs.clone())?;
        let loss = ys.sub(sample_ys.clone())?.square()?.sum(&[0])?;
        let mut grads = loss.backward()?;
        opt.backward_step(&mut grads, &device)?;
        // device.try_gpu().unwrap().mark_step().unwrap();
        let b = b.as_tensor().to(&Device::CPU)?;
        let w = w.as_tensor().to(&Device::CPU)?;
        println!("b: {:?}, w: {:?}", b.to_vec::<f32>(), w.to_vec::<f32>());
        let loss_cpu = loss.clone().to(&Device::CPU)?;
        let loss_vec = loss_cpu.to_vec::<f32>()?;
        println!("loss: {:?}", loss_vec[0]);
    }

    let b = b.as_tensor().to(&Device::CPU)?;
    let w = w.as_tensor().to(&Device::CPU)?;
    println!("b: {:?}, w: {:?}", b.to_vec::<f32>(), w.to_vec::<f32>());
    assert_eq!(to_vec0_round(&b, 4)?, 0.7872);
    assert_eq!(to_vec1_round(&w, 4)?, &[2.7257, 0.7097]);
    Ok(())
}

#[test]
fn sgd_linear_regression() -> anyhow::Result<()> {
    fn optimizer(vars: Vec<(Option<String>, Var)>) -> anyhow::Result<SGD> {
        SGD::new(vars, 0.001)
    }
    run_linear_regression(optimizer)
}

#[test]
fn adamw_linear_regression() -> anyhow::Result<()> {
    fn optimizer(vars: Vec<(Option<String>, Var)>) -> anyhow::Result<AdamW> {
        let params = ParamsAdamW {
            lr: 0.5,
            ..Default::default()
        };
        AdamW::new(vars, params)
    }
    run_linear_regression(optimizer)
}

fn gradient_descent(optimizer: OptimizerFactory<impl Optimizer>) -> anyhow::Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();

    let device = Device::request_device(DeviceRequest::GPU).unwrap();
    let target = Tensor::from_data(vec![5.0], shape![1], Device::CPU).to(&device)?;

    // Initialize variable at 0.0 (shape is scalar)
    let w = Var::from_tensor(&Tensor::from_data(vec![0.0], shape![1], Device::CPU).to(&device)?)?;

    let mut opt = optimizer(vec![(Some("w".into()), w.clone())])?;

    for step in 0..100 {
        // Compute loss = (w - target)^2
        let loss = w.as_tensor().clone().sub(target.clone())?.square()?;

        // Backpropagate
        let mut grads = loss.backward()?;
        opt.backward_step(&mut grads, &device)?;

        // Print debug info
        let current_w = w.as_tensor().to(&Device::CPU)?.to_vec::<f32>()?;
        let current_loss = loss.to(&Device::CPU)?.to_vec::<f32>()?;
        #[cfg(feature = "plotting")]
        println!(
            "Step {step}: w = {:.4}, loss = {:.4} (fmt: {})",
            current_w[0],
            current_loss[0],
            w.as_tensor().to(&Device::CPU)?.plot_fmt()
        );
        println!(
            "Step {step}: w = {:.4}, loss = {:.4}",
            current_w[0], current_loss[0],
        );
    }

    let final_w = w.as_tensor().to(&Device::CPU)?.to_vec::<f32>()?;
    assert!(
        (final_w[0] - target.to(&Device::CPU)?.to_vec::<f32>()?[0]).abs() < 0.1,
        "Final w should be close to 5.0"
    );
    Ok(())
}

#[test]
fn sgd_gradient_descent() -> anyhow::Result<()> {
    fn optimizer(vars: Vec<(Option<String>, Var)>) -> anyhow::Result<SGD> {
        SGD::new(vars, 0.1)
    }
    gradient_descent(optimizer)
}

#[test]
fn adamw_gradient_descent() -> anyhow::Result<()> {
    fn optimizer(vars: Vec<(Option<String>, Var)>) -> anyhow::Result<AdamW> {
        let params = ParamsAdamW {
            lr: 0.2,
            ..Default::default()
        };
        AdamW::new(vars, params)
    }
    gradient_descent(optimizer)
}

#[test]
fn test_intermediate() -> anyhow::Result<()> {
    let device = Device::request_device(DeviceRequest::GPU).unwrap();
    let w_gen = Tensor::from_data(vec![3f32, 1.], shape![1, 2], Device::CPU).to(&device)?;
    let b_gen = Tensor::from_data(vec![-2f32], shape![1, 1], Device::CPU).to(&device)?;
    let gen = Linear::new(w_gen.clone(), Some(b_gen.clone()));
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
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));

    let ys = lin.schedule(sample_xs.clone())?;
    let loss = ys.sub(sample_ys.clone())?.square()?;

    // Print loss
    println!("loss: {:?}", loss.to(&Device::CPU)?.to_vec::<f32>()?);

    let gen_grads = loss.backward()?;
    for (i, (_id, g)) in gen_grads.iter().enumerate() {
        let g_clone = g.clone();
        println!(
            "gen_grads[{}]: (op {:?}) {:?}",
            i,
            g_clone.clone().op().name(),
            g_clone.to(&Device::CPU)?.to_vec::<f32>()?
        );
    }
    Ok(())
}
