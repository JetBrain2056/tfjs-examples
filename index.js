const URL      = "./model/";
const imageURL = './images/6.jpg';
let model;

let info       = document.getElementById('info');
let prediction = document.getElementById('prediction');
let progress   = document.querySelector('progress');
let image      = document.getElementById('image');

image.src = imageURL;

const trainButton = document.getElementById('train');
trainButton.onclick = async function() {
  
  const input  = tf.tensor2d(TRAINING_DATA.inputs);
  const output = tf.oneHot(tf.tensor1d(TRAINING_DATA.outputs, 'int32'), 10);

  model = tf.sequential();

  model.add(tf.layers.dense({inputShape: [784], units: 64}));
  model.add(tf.layers.dense({units: 16}));
  model.add(tf.layers.dense({units: 10, activation:'softmax'}));
  
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });  

  model.summary();

  info.innerText = 'Training model. Please wait...';

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

}
const saveButton = document.getElementById('save');
saveButton.onclick = async function saveModel() {
  console.log('SaveModel...');
  
  try {
    await model.save('downloads://model');
  } catch(err) {
    console.log(err);
  }
  
}

const runButton = document.getElementById('run');
runButton.onclick = function runPredict() {
  console.log("run predict...");

  console.log(TRAINING_DATA.inputs[8]);
  console.log(TRAINING_DATA.outputs.length);

  const bufferT   = tf.browser.fromPixels(image);
  const expandedT = tf.image.resizeBilinear(bufferT, [28, 28]);
  const imageT    = tf.cast(tf.expandDims(expandedT), 'int32');
  const newImageT = convertImage(imageT.dataSync());

  console.log(imageT.dataSync()); 
  console.log(newImageT);  

  // i = Math.floor(Math.random() * TRAINING_DATA.inputs.length);
  // const testxs = tf.tensor1d(TRAINING_DATA.inputs[i]).expandDims();

  const testxs = tf.tensor1d(newImageT).expandDims();
  // console.log(testxs);
         
  output = model.predict(testxs).dataSync();   

  arr    = Array.from(output);
  number = arr.indexOf(Math.max(...arr));
  console.log('number', number);
  prediction.innerText = number;

  // drawImage(TRAINING_DATA.inputs[i]);
  drawImage(newImageT);
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
function convertImage(imageData) {
  let digit = [];
  
  for (let ii = 0; ii < 784; ii++) {
    for (let i = 0; i < imageData.length; i++) {
      digit[ii] = Math.ceil((255-imageData[ii*3])/255*100)/100;           
    }
  }
  return digit;
}

// Load the image model 
async function init() {
    const modelURL    = URL + "model.json";
  
    // model = await tf.loadLayersModel(modelURL);
    // model.summary();
}

window.onload = () => init(); 
