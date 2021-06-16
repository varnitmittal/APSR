import React from 'react';
import './App.css';

import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import DispArea from './components/DispArea.js';

import './styling/css/icomoon.css';
//import './styling/css/bootstrap.css';
import './styling/fonts/flaticon/font/flaticon.css';
import './styling/css/style.css';

function App() {
  return (
    <div className="App">
      <Navbar />
      <div className="main">
        <Sidebar />
        <DispArea />
      </div>
    </div>
  );
}

export default App;
