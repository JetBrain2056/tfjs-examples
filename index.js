const URL      = "./model/";
const imageURL = './images/1.png';

let model;

let info       = document.getElementById('info');
let prediction = document.getElementById('prediction');
let progress   = document.querySelector('progress');
let image      = document.getElementById('image');

// image.src = imageURL;

const trainButton = document.getElementById('train');
trainButton.onclick = async function() {

  //convert to tensor 
  //const bufferT   = tf.browser.fromPixels(image);
  // const expandedT = await tf.image.resizeBilinear(bufferT, [224, 224]);
  //const tensorImg = tf.cast(tf.expandDims(bufferT), 'int32');

  //console.log(tensorImg);

  const input  = tf.tensor2d(TRAINING_DATA.inputs);
  const output = tf.oneHot(tf.tensor1d(TRAINING_DATA.outputs, 'int32'), 10);

  model = await tf.sequential();
    
  // model.add(tf.layers.conv2d({
  //   inputShape: [image.height, image.width, 3],
  //   kernelSize: 3,
  //   filters: 16,
  //   activation: 'relu'
  // }));

  // model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  // model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  // model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  // model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  // model.add(tf.layers.flatten({}));
  model.add(tf.layers.dense({inputShape: [784], units: 64}));
  model.add(tf.layers.dense({units: 16}));
  model.add(tf.layers.dense({units: 10, activation:'softmax'}));
  
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });  

  model.summary();

  info.innerText = 'Training model. Please wait...';
  // progress.style.display = 'block';

  const epochs = 50;
  await model.fit(input, output, {
    epochs          : epochs, 
    batchSize       : 512,     
    shuffle         : true,
    callbacks       : { onEpochEnd: async (epoch,logs) => {
      progress.value = epoch/(epochs-1)*100;
      console.log('Epoch', epoch, logs)
    }}
  });

  info.innerText = 'Model succesfully trained.';
  // progress.style.display = 'none';

  // await model.save('./model/');

  // await runPredict();

}

const runButton = document.getElementById('run');
runButton.onclick = async function runPredict() {
  console.log("run predict...");

  i = Math.floor(Math.random() * TRAINING_DATA.inputs.length);

  const testxs = await tf.tensor1d(TRAINING_DATA.inputs[i]).expandDims();

  let result = tf.tidy(() => {
    data = model.predict(testxs);  
    return data.argMax(-1);
  })
  result.array().then(number => {  
    console.log('number', number);
    prediction.innerText = number;
    drawImage(TRAINING_DATA.inputs[i]);
  })
}

function drawImage(digit) {
  let context   = document.querySelector('canvas').getContext('2d');
  let imageData = context.getImageData(0, 0, 28, 28);
  for (let i = 0; i < digit.length; i++){
    imageData.data[i * 4] = digit[i] * 255;      
    imageData.data[i * 4 + 1] = digit[i] * 255; 
    imageData.data[i * 4 + 2] = digit[i] * 255; 
    imageData.data[i * 4 + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
}

async function test() {

  const bufferT   = await tf.browser.fromPixels(image);
  const expandedT = await tf.image.resizeBilinear(bufferT, [224, 224]);
  const imageT    = await tf.cast(await tf.expandDims(expandedT), 'int32');
  
  console.log('loaded image:', imageT['file'], 'width:', imageT.shape[2], 'height:', imageT.shape[1]);

  // Get the output tensors.
  let result = await model.predictOnBatch(imageT);
  
  console.log('Predictions: ', result);
  console.log('data: ', result.data());

  const c = document.getElementById('canvas');
  const context = c.getContext('2d');
  context.drawImage(image, 0, 0);
  context.font = '12px Arial';

  console.log('number of detections: ', result.length);
  for (let i = 0; i < result.length; i++) {
    context.beginPath();
    context.rect(...result[i].bbox);
    context.lineWidth   = 2;
    context.strokeStyle = 'green';
    context.fillStyle   = 'green';
    context.stroke();
    context.fillText(
        result[i].score.toFixed(3) + ' ' + result[i].class, result[i].bbox[0],
        result[i].bbox[1] > 10 ? result[i].bbox[1] - 5 : 10);
  }
}

// Load the image model 
async function init() {
    const modelURL    = URL + "model.json";
    const metadataURL = URL + "metadata.json";
    
    // model = await tf.loadLayersModel(modelURL);
    // model.summary();
}

window.onload = () => init(); 