const {publicEncrypt,privateDecrypt} =require('crypto')

const {publicKey, privateKey}= require('./keypair')

const message = 'Welcome To Entanglia ! :)'

const encryptedData = publicEncrypt(
    publicKey,
    Buffer.from(message)
)

console.log(encryptedData.toString('hex'))

const decryptedData = privateDecrypt(privateDecrypt,encryptedData)

console.log(decryptedData)