import { initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAjI3IDARlu7hDwXDIxdbjpF7JIq2aMSnw",
  authDomain: "classification-techniques.firebaseapp.com",
  databaseURL: "https://classification-techniques-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "classification-techniques",
  storageBucket: "classification-techniques.firebasestorage.app",
  messagingSenderId: "900851830642",
  appId: "1:900851830642:web:20986e9c36e8859aec57eb",
  measurementId: "G-LQ8HHC4QKW"
};

const app = initializeApp(firebaseConfig);
export const db = getDatabase(app);
export const configured = true;

// Only URLs that include ?key=<this value> can broadcast as presenter.
// Change this to any secret phrase before sharing your audience link.
export const PRESENTER_KEY = "mypin";
