const express = require('express')
const session = require('express-session')
const MongoStore = require('connect-mongo')
const path = require('path')
const passport = require('passport')
const { promisify } = require('es6-promisify')
const flash = require('connect-flash')
const cookieParser = require('cookie-parser')
const bodyParser = require('body-parser')
const routes = require('./routes/index')
const helpers = require('./helpers')
const errorHandlers = require('./handlers/errorHandlers')
require('./handlers/passport')

// create express app 
const app = express()

if (process.env.NODE_ENV === 'production') {
    app.use((req, res, next) => {
        if (req.header('x-forwarded-proto') !== 'https')
            res.redirect(`https://${req.header('host')}${req.url}`)
        else
            next()
    })
}

// view engine setup
app.use(express.static('public'))
app.set('views', path.join(__dirname, 'views'))
app.set('view engine', 'pug')

// Takes the raw requests and turns them into usable properties on req.body
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }))

// Exposes a bunch of methods for validating data. Used heavily on userController.validateRegister
// app.use(buildCheckFunction())
// app.use(buildSanitizeFunction())

// populates req.cookies with any cookies that came along with the request
app.use(cookieParser())

// use CORS

// Sessions allow us to store data on visitors from request to request
// This keeps users logged in and allows us to send flash messages
app.use(session({
    secret: process.env.SECRET,
    key: process.env.KEY,
    resave: false,
    saveUninitialized: false,
    store: MongoStore.create({
        mongoUrl: process.env.DATABASE
    })
}))

// Passport JS is what we use to handle our logins
app.use(passport.initialize())
app.use(passport.session())

// use flash to pass message to the next page the user request
app.use(flash())

// pass variables to all of the templates and all request
app.use(async(req, res, next) => {
    res.locals.h = helpers
    res.locals.flashes = req.flash()
    res.locals.user = req.user || null
    res.locals.currentPath = req.path
    next()
})

// promisify some callback based APIs
app.use((req, res, next) => {
    req.login = promisify(req.login, req)
    next()
})

app.use('/', routes)

// if that above routes didnt work, 404 them and forward to error handler
app.use(errorHandlers.notFound)

// one of the error handlers will see if these errors are just validation errors
app.use(errorHandlers.flashValidationErrors)

// otherwise 
if (app.get('env') === 'development') {
    /* Development Error Handler - Prints stack trace */
    app.use(errorHandlers.developmentErrors)
}

// production error handler
app.use(errorHandlers.productionErrors)

module.exports = app