const express = require('express')
const controller = require('../controllers/controller')

const router = express.Router()

router.get('/', controller.home)
router.get('/presentation', controller.presentation)

module.exports = router