const {createSign, createVerify} = require('crypto'); 

const {publicKey, privateKey} = require('./keypair')

const message = 'This must be signed'; 
const signer = createSign('rsa-sha256');


const signature = signer .sign(privateKey, 'hex');

const verifier = createVerify('rsa-sha256')

verifier.update(message);
const isVerified = verifier.verify(publicKey)