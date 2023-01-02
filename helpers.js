const fs = require('fs')

exports.moment = require('moment')

// helps in debugging when using templates
exports.dump = (obj) => JSON.stringify(obj, null, 2)

// Icons directory (inserting an svg)
exports.icon = (name) => fs.readFileSync(`./public/images/icons/${name}.svg`)

// Images directory (inserting an image)
exports.image = (name) => fs.readFileSync(`./public/images/${name}.png`)

// site info
exports.siteName = `Crystal Blue Tech`

// exports.cart = commerce.cart.retrieve().then((res => { return res.line_items.length })).catch(error => console.log(error))

exports.menu = [
    { title: 'Men', link: '/categories/men' },
    { title: 'Women', link: '/categories/women' },
    { title: 'About', link: '/about' }
]