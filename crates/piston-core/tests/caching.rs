use piston::{Device, DeviceRequest, Tensor, TensorOptions};

// TODO(vinhowe): Remove or motivate this test
#[test]
fn test_simple_caching() -> anyhow::Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();

    let device = Device::request_device(DeviceRequest::GPU).unwrap();
    let t1 = Tensor::from_data(vec![1f32], 1, TensorOptions::new())?
        .to(&device)?
        .square()?;
    println!("t1: {:?}", t1.to(&Device::CPU)?.to_vec::<f32>()?);

    assert_eq!(t1.to(&Device::CPU)?.to_vec::<f32>()?, vec![1f32]);

    let t2 = Tensor::from_data(vec![2f32], 1, TensorOptions::new())?
        .to(&device)?
        .square()?;
    println!("t2: {:?}", t2.to(&Device::CPU)?.to_vec::<f32>()?);

    assert_eq!(t2.to(&Device::CPU)?.to_vec::<f32>()?, vec![4f32]);

    let t3 = Tensor::from_data(vec![3f32], 1, TensorOptions::new())?
        .to(&device)?
        .square()?;
    println!("t3: {:?}", t3.to(&Device::CPU)?.to_vec::<f32>()?);

    assert_eq!(t3.to(&Device::CPU)?.to_vec::<f32>()?, vec![9f32]);

    let t4 = Tensor::from_data(vec![4f32], 1, TensorOptions::new())?
        .to(&device)?
        .square()?;
    println!("t4: {:?}", t4.to(&Device::CPU)?.to_vec::<f32>()?);

    assert_eq!(t4.to(&Device::CPU)?.to_vec::<f32>()?, vec![16f32]);

    let t5 = Tensor::from_data(vec![5f32], 1, TensorOptions::new())?
        .to(&device)?
        .square()?;
    println!("t5: {:?}", t5.to(&Device::CPU)?.to_vec::<f32>()?);

    assert_eq!(t5.to(&Device::CPU)?.to_vec::<f32>()?, vec![25f32]);

    Ok(())
}
