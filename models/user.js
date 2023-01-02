const mongoose = require('mongoose')
const validator = require('validator')
const passportLocalMongoose = require('passport-local-mongoose')
const mongodbErrorHandler = require('mongoose-mongodb-errors')

const Schema = mongoose.Schema

const userSchema = new Schema({
    name: {
        type: String,
        required: 'You must provide a name',
        trim: true
    },
    email: {
        type: String,
        required: 'You must provide an email',
        trim: true,
        lowercase: true,
        validate: [validator.default.isEmail, 'Invalid email address']
    }
})

userSchema.plugin(passportLocalMongoose, {
    usernameField: 'email'
})
userSchema.plugin(mongodbErrorHandler)

module.exports = mongoose.model('User', userSchema)