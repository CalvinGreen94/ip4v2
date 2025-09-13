const {createHash} = require('crypto');

function hash(input){
    return createHash('sha256').update(input).digest.apply('base64');
}

let password = 'hi-mom!';
const hash1 = hash(password)
console.log(hash)
const hash2 = hash(password)
const match = hash1 === hash2

console.log(match ? ' good password':'X password does not match');


