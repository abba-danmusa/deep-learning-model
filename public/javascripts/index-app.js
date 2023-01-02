import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

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

function createModel() {
    // create a sequential model
    const model = tf.sequential()

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

function predict() {
    const predictionInput = parseInt(document.querySelector('#sqrfoot').value)
    if (isNaN(predictionInput)) {
        alert('Square foot must be a number, Please enter a valid number')
    } else {
        tf.tidy(() => {
            const inputTensor = tf.tensor1d([predictionInput])
            const normalisedInput = normalise(inputTensor, normalisedFeature.min, normalisedFeature.max)
            const normalisedOutputTensor = model.predict(normalisedInput.tensor)
            const outputTensor = denormalise(normalisedOutputTensor, normalisedLabel.min, normalisedLabel.max)
            const outputValue = outputTensor.dataSync()[0]

            // show the predicted price
            alert(`The predicted price is ${outputValue}`)
        })
    }

}

// setTimeout(() => {
//     const submitBtn = document.querySelector('#submitBtn')
//     submitBtn.addEventListener('click', predict)
// }, 5000)

let normalisedFeature, normalisedLabel, points, model

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

    // split the normalised tensors [feature and label] into training and testing datasets
    const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalisedFeature.tensor, 2)
    const [trainingLabelTensor, testingLabelTensor] = tf.split(normalisedLabel.tensor, 2)

    model = createModel()
    tfvis.show.modelSummary({ name: 'Model Summary' }, model)
    const layer = model.getLayer(undefined, 0) // get the model layer 
    tfvis.show.layer({ name: 'Layer 1' }, layer)
    await plotPredictionLine()

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor)

    // await model.save('/model/model.json')

    // console.log(result)
    const trainingLoss = result.history.loss.pop()
    console.log(`Training set loss: ${trainingLoss}`)

    const validationLoss = result.history.val_loss.pop()
    console.log(`Validation loss ${validationLoss}`)

    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor)
    const loss = await lossTensor.dataSync()
    console.log(`Testing set loss ${loss}`)

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