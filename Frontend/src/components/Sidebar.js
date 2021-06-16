import React, { Component } from 'react';
import SidebarFooter from './SidebarFooter';
import '../styling/css/sidebar.css'

export default class Sidebar extends Component {
    render() {
        return (
            <div className="mycss-sbar" /* style={{border: '4px solid red'}} */>
                <div className="mycss-sbar-content" /* style={{border: '4px solid green'}} */>
                    <h2 id=""><a href="/">Recognizes empty parking slots from images using deep learning convolutional neural network</a></h2>
                    <h3 id="mycss-sbar-text">
                    Recognizes empty parking spaces from images using deep learning convolutional neural network model trained on custom dataset.<br /> 
              Entire frontend is responsive and built with React.<br /> The images are processed on server-side and sent back as API response.<br /> 
              The API is built with Flask served by Gunicorn WSGI server and deployed to Heroku.
                    </h3>
                    <h4 id="mycss-sbar-text"> The Developers: 
                        <ul>
                            <li>Varnit Mittal</li>
                            <li>Divyam Goyal</li>
                            <li>Pulkit Jain</li>
                            <li>Albin Joseph</li>
                        </ul>
                    </h4>
                </div><SidebarFooter />
           </div>
        )
    }
}
