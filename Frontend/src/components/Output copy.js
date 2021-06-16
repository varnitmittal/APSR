import React, { Component } from 'react';
import '../styling/css/output.css'

export default class Output extends Component {
    constructor(props){
        super(props);
        this.state = {
            /* imstr: this.props.imstrProp, */
            status: this.props.statusProp,
            detections: this.props.detectionsProp
        }
        console.log("API response received: "+this.state.detections)
    }

    render() {
        return (
            <div>
{/*                 <div>
                    <p style={{ color:'#fff', fontSize:'1rem' }}>
                        <a href={`data:image/jpeg;base64,${this.state.imstr}`} download='image_after_detection.jpg'>
                            Download This Image
                        </a>
                    </p>
                </div> */}
{/*                 <div>
                    <img className="mycss-o-img" src={`data:image/jpeg;base64,${this.state.imstr}`}></img>
                </div> */}
                <div>
                     <ul className='mycss-list'>
                        {this.state.detections.length
                        ?
                            /* this.state.detections.map((value, index) => {
                                return  <li key={index}>
                                            {value.object} - {value.probability.toFixed(2)}%
                                        </li>
                            }) */
                            <h1 style={{ color: '#ffffff' }}>The Plate:  {this.state.detections}</h1>
                        :
                            <div>
                                <p style={{ color:'#fff !important ', fontSize:'1rem' }}>
                                    Couldn't recognize the plate.
                                </p>
                            </div>
                        }
                    </ul>
                </div>
                <div>
                    <p style={{ color:'#323235 !important ', fontSize:'1rem' }}>
                        <a href='/'>
                            Try Another Image
                        </a>
                    </p>
                </div>
            </div>
        )
    }
}
