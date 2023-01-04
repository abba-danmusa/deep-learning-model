import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

const storageID = `kc-house-price-regression`

const toggleButton = document.getElementById('toggle-button')
const trainButton = document.getElementById('train-button')
const testButton = document.getElementById('test-button')
const loadButton = document.getElementById('load-button')
const saveButton = document.getElementById('save-button')
const predictButton = document.getElementById('predict-button')

toggleButton.addEventListener('click', toggleVisor)
trainButton.addEventListener('click', train)
testButton.addEventListener('click', test)
loadButton.addEventListener('click', load)
saveButton.addEventListener('click', save)
predictButton.addEventListener('click', predict)

function plot(pointsArray, featureName, predictedPointsArray = null) {
    // const values = [pointsArray.slice(0, 1000)]
    const values = [pointsArray]
    const series = ['original']

    if (Array.isArray(predictedPointsArray)) {
        values.push(predictedPointsArray)
        series.push('predicted')
    }

    tfvis.render.scatterplot({ name: `${featureName} vs House price` }, { values: [pointsArray], series: ["Original Data"] }, { xLabel: featureName, yLabel: "Price" })
    tfvis.render.scatterplot({ name: `${featureName} vs House price` }, { values, series }, { xLabel: featureName, yLabel: "Price" })

}

async function plotPredictionLine() {
    const [xs, ys] = tf.tidy(() => {
        const normalizedXs = tf.linspace(0, 1, 100)
        const normalizedYs = model.predict(normalizedXs.reshape([100, 1]))

        const xs = denormalise(normalizedXs, normalisedFeature.min, normalisedFeature.max)
        const ys = denormalise(normalizedYs, normalisedLabel.min, normalisedLabel.max)

        return [xs.dataSync(), ys.dataSync()]
    })
    const predictedPoints = Array.from(xs).map((val, index) => {
        return { x: val, y: ys[index] }
    })
    plot(points, 'Square feet', predictedPoints)
}

// normalises a given tensor
function normalise(tensor, previousMin = null, previousMax = null) {

    // get the min and max of values of the tensor
    const max = previousMax || tensor.max()
    const min = previousMin || tensor.min()

    // subtract the min from the tensor, divide by the max subtracted from the min
    const normalisedTensor = tensor.sub(min).div(max.sub(min))
    return {
        tensor: normalisedTensor,
        min,
        max
    }

}

// denormalises a tensor
function denormalise(tensor, min, max) {
    const denormaliseTensor = tensor.mul(max.sub(min)).add(min)
    return denormaliseTensor
}

let model = null

function createModel() {
    // create a sequential model
    model = tf.sequential()

    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
        inputDim: 1
    }))

    // model.add(tf.layers.dense({
    //     units: 10,
    //     useBias: true,
    //     activation: 'sigmoid',
    //     inputDim: 1
    // }))
    // model.add(tf.layers.dense({
    //     units: 10,
    //     useBias: true,
    //     activation: 'sigmoid'
    // }))
    // model.add(tf.layers.dense({
    //     units: 1,
    //     useBias: true,
    //     activation: 'sigmoid'
    // }))

    // const optimizer = tf.train.sgd(0.1)

    const optimizer = tf.train.adam()

    model.compile({
        loss: 'meanSquaredError',
        optimizer
    })

    return model
}

async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
    // plot on the visor on each epoch and batch end
    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks({ name: "Training Performance" }, ['loss'])
    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        // batchSize: 512,
        // epochs: 1,
        epochs: 20,
        validationSplit: .2,
        callbacks: {
            onEpochEnd,
            onEpochBegin: async function() {
                await plotPredictionLine()
                const layer = model.getLayer(undefined, 0) // get the model layer 
                tfvis.show.layer({ name: 'Layer 1' }, layer)
            }

            // onBatchEnd
        }
    })
}

let normalisedFeature, points
let normalisedLabel
let trainingFeatureTensor, trainingLabelTensor
let testingFeatureTensor, testingLabelTensor

async function dataSet() {

    // import data from a csv file
    const dataset = tf.data.csv('../dataset/kc_house_data.csv')

    // extract x and y values to plot
    const pointsDataset = dataset.map((record) => ({
        x: record.sqft_living,
        y: record.price
    }))

    points = await pointsDataset.toArray()

    // shuffle the data...
    // ...if points array is odd
    if (points.length % 2 !== 0) points.pop() // pop one element from the array
    tf.util.shuffle(points) // shuffle the points array

    // plot the data on the visor
    plot(points, "Square feet")

    // get the x values and y values [feature values and label values] from the points array
    const featureValues = points.map(p => p.x)
    const labelValues = points.map(p => p.y)

    // create a two dimentional tensors for the feature values and the label values [inputs and outputs]
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])
    featureTensor.print()
    labelTensor.print()

    // normalise features and labels
    normalisedFeature = normalise(featureTensor)
    normalisedLabel = normalise(labelTensor)
    featureTensor.dispose();
    labelTensor.dispose();

    // split the normalised tensors [feature and label] into training and testing datasets
    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalisedFeature.tensor, 2)

    [trainingLabelTensor, testingLabelTensor] = tf.split(normalisedLabel.tensor, 2)


    // Update status and enable train button
    document.getElementById("model-status").innerHTML = "No model trained"
    document.getElementById("train-button").removeAttribute("disabled")
    document.getElementById("load-button").removeAttribute("disabled")
}

function predict() {

    const predictionInput = parseInt(document.getElementById("prediction-input").value)

    if (isNaN(predictionInput)) {
        alert("Please enter a valid number")
    } else {
        tf.tidy(() => {
            const inputTensor = tf.tensor1d([predictionInput])
            const normalisedInput = normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
            const normalisedOutputTensor = model.predict(normalisedInput.tensor)
            const outputTensor = denormalise(normalisedOutputTensor, normalisedLabel.min, normalisedLabel.max)
            const outputValue = outputTensor.dataSync()[0]
            document.getElementById("prediction-output").innerHTML = `The predicted house price is: <br />` +
                `<span style="font-size: 2em">\$${(outputValue/1000).toFixed(0)*1000}</span>`
        })
    }
}

async function load() {
    const storageKey = `localstorage://${storageID}`;
    const models = await tf.io.listModels();
    const modelInfo = models[storageKey];
    if (modelInfo) {
        model = await tf.loadLayersModel(storageKey);

        tfvis.show.modelSummary({ name: "Model summary" }, model);
        const layer = model.getLayer(undefined, 0);
        tfvis.show.layer({ name: "Layer 1" }, layer);

        document.getElementById("model-status").innerHTML = `Trained (saved ${modelInfo.dateSaved})`;
        document.getElementById("predict-button").removeAttribute("disabled");
    } else {
        alert("Could not load: no saved model found");
    }
    document.getElementById("predict-button").removeAttribute("disabled")
    await plotPredictionLine()
}

async function save() {
    const saveResults = await model.save(`localstorage://${storageID}`)
    document.getElementById("model-status").innerHTML = `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`
}

async function test() {

    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor)
    const loss = await lossTensor.dataSync()
    console.log(`Testing set loss ${loss}`)

    document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss.toPrecision(5)}`
}

async function train() {

    // Disable all buttons and update status
    ["train", "test", "load", "predict", "save"].forEach(id => {
        document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
    });
    document.getElementById("model-status").innerHTML = "Training..."

    model = createModel()
    tfvis.show.modelSummary({ name: 'Model Summary' }, model)
    const layer = model.getLayer(undefined, 0) // get the model layer 
    tfvis.show.layer({ name: 'Layer 1' }, layer)
    await plotPredictionLine()

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor)

    // console.log(result)
    const trainingLoss = result.history.loss.pop()
    console.log(`Training set loss: ${trainingLoss}`)

    const validationLoss = result.history.val_loss.pop()
    console.log(`Validation loss ${validationLoss}`)

    document.getElementById("model-status").innerHTML = `Trained (unsaved)\nLoss: ${trainingLoss.toPrecision(5)}\nValidation loss: ${validationLoss.toPrecision(5)}`;

    document.getElementById("test-button").removeAttribute("disabled")
    document.getElementById("save-button").removeAttribute("disabled")
    document.getElementById("predict-button").removeAttribute("disabled")
}

async function toggleVisor() {
    tfvis.visor().toggle()
}

export async function plotParams(weight, bias) {
    model.getLayer(null, 0).setWeights([
        tf.tensor2d([
            [weight]
        ]), // kernel (input multiplier)
        tf.tensor1d([bias]) // Bias
    ])
    await plotPredictionLine()
    const layer = model.getLayer(undefined, 0) // get the model layer 
    tfvis.show.layer({ name: 'Layer 1' }, layer)
}

dataSet()