import * as tf from '@tensorflow/tfjs'
import * as DICTIONARY from '/dictionary.js'

const MODEL_JSON_URL = '../model/model.json'
const SPAM_THRESHOLD = .75

// The number of input elements the ML Model is expecting.
const ENCODING_LENGTH = 20

let model = undefined

async function loadAndPredict(inputTensor) {
    // Load the model.json and binary files you hosted. Note this is 
    // an asynchronous operation so you use the await keyword
    if (model === undefined) {
        model = await tf.loadLayersModel(MODEL_JSON_URL);
    }

    // Once model has loaded you can call model.predict and pass to it
    // an input in the form of a Tensor. You can then store the result.
    let results = await model.predict(inputTensor);

    // Print the result to the console for us to inspect.
    results.print();

    // TODO: Add extra logic here later to do something useful
}

loadAndPredict(
    tf.tensor([
        [1, 3, 12, 18, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
)

/** 
 * Function that takes an array of words, converts words to tokens,
 * and then returns a Tensor representation of the tokenization that
 * can be used as input to the machine learning model.
 */
function tokenize(wordArray) {
  // Always start with the START token.
  let returnArray = [DICTIONARY.START];
  
  // Loop through the words in the sentence you want to encode.
  // If word is found in dictionary, add that number else
  // you add the UNKNOWN token.
  for (var i = 0; i < wordArray.length; i++) {
    let encoding = DICTIONARY.LOOKUP[wordArray[i]];
    returnArray.push(encoding === undefined ? DICTIONARY.UNKNOWN : encoding);
  }
  
  // Finally if the number of words was < the minimum encoding length
  // minus 1 (due to the start token), fill the rest with PAD tokens.
  while (i < ENCODING_LENGTH - 1) {
    returnArray.push(DICTIONARY.PAD);
    i++;
  }
  
  // Log the result to see what you made.
  console.log([returnArray]);
  
  // Convert to a TensorFlow Tensor and return that.
  return tf.tensor([returnArray]);
}

function plotClasses(pointsArray, classKey) {
    const allSeries = {}

    // add each class to a series
    pointsArray.forEach(p => {
        // add each point to the serires for the class it is in
        const seriesName = `${classKey}: ${p.class}`
        let series = allSeries[seriesName]
        if (!series) {
            series = []
            allSeries[seriesName] = series
        }
        series.push(p)
    })

    tfvis.render.scatterplot({ name: 'Email vs prediction' }, {
        values: Object.values(allSeries),
        series: Object.keys(allSeries)
    }, {
        xLabel: 'Square feet',
        yLabel: 'Price'
    })
}