const nodeMailer = require('nodemailer')
const pug = require('pug')
const juice = require('juice')
const htmlToText = require('html-to-text')
const { promisify } = require('es6-promisify')

const transport = nodeMailer.createTransport({
    host: process.env.MAIL_HOST,
    port: process.env.MAIL_PORT,
    auth: {
        user: process.env.MAIL_USER,
        pass: process.env.MAIL_PASS
    }
})

exports.send = async(options) => {
    const mailOptions = {
        from: `Abba Danmusa <noreply@abbadanmusa.dev>`,
        to: options.email,
        subject: options.subject,
        html: 'to fill in later',
        text: 'to fill in later'
    }

    // const sendMail = promisify(transport.sendMail, transport)
    return transport.sendMail(mailOptions)

}

// transport.sendMail({
//     from: 'Abba Danmusa <contact-me@abbadanmusa.com>',
//     to: 'me@example.com',
//     subject: 'just trying things out',
//     html: 'thank you, <strong>love you</strong>',
//     text: 'thank you **love you**'
// })