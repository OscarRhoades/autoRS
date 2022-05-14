
use mnist::*;
use ndarray::prelude::*;

use image::RgbImage;


fn array_to_image(arr: Array3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (height, width, _) = arr.dim();
    let raw = arr.into_raw_vec();

    RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}




pub fn data() -> (Array3<f32>, Array2<f32>){

   // Deconstruct the returned Mnist struct.
   let Mnist {
       trn_img,
       trn_lbl,
       tst_img,
       tst_lbl,
       ..
   } = MnistBuilder::new()
       .label_format_digit()
       .training_set_length(50_000)
       .validation_set_length(10_000)
       .test_set_length(10_000)
       .finalize();

   
   // Can use an Array2 or Array3 here (Array3 for visualization)
   let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
       .expect("Error converting images to Array3 struct")
       .map(|x| *x as f32 / 256.0);


   // Convert the returned Mnist struct to Array2 format
   let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
       .expect("Error converting training labels to Array2 struct")
       .map(|x| *x as f32);


    (train_data, train_labels)
}



pub fn generate(label: i32) -> (Array2<f32>, f32){

    let gen = data();
    let data = gen.0.slice(s![label,.., ..]);
    let tag = gen.1.slice(s![label, ..])[0];
    
   (data.map(|x| *x as f32), tag)
     
}



pub fn show(label: i32){

    let gen = data();
    let data = gen.0.slice(s![label, .., ..]);

    let mut array: Array3<u8> = Array3::zeros((28, 28, 3));

    for ((x, y, z), v) in array.indexed_iter_mut() {
        *v = match z {
            _ => (data[(x,y)] * 255.0) as u8
        };
    }


    println!("The first digit is a {:?}", gen.1.slice(s![label, ..])[0] );
    
    let image = array_to_image(array);
    image.save("out.png");
}