const { EnvironmentPlugin } = require('webpack')
const path = require('path')

const javascript = {
    test: /\.m?js$/,
    exclude: /(node_modules|bower_components)/,
    use: [{
        loader: 'babel-loader'
    }],
}

// dotenv.config({ path: '.env' })
module.exports = {

    node: {
        global: true,
        __filename: true,
        __dirname: true
    },
    entry: {
        // App: './public/javascripts/index-app.js',
        App: './public/javascripts/index-app.js',
    },
    output: {
        path: path.resolve(__dirname, 'public', 'dist'),
        // filename: '[name].bundle.js',
        filename: '[name].bundle.js',
        libraryTarget: 'var',
        library: 'webpack'
    },
    module: {
        rules: [javascript]
    },
    resolve: {
        modules: ['node_modules']
    },
    mode: 'development',
    watch: true

}

process.noDeprecation = true