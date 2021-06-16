import React, { Component } from 'react';
import Input from '../components/Input';
import config from '../config/config';
import axios from 'axios';
import Output from './Output';

import { trackPromise } from 'react-promise-tracker';

export default class DispArea extends Component {
    constructor(props){
        super(props);
        this.hitAPI = this.hitAPI.bind(this);
        this.stringAdjust = this.stringAdjust.bind(this);
        this.state = {
            inputTaken: false
        }
    }

    componentDidMount() {
       /*  axios.get(config.apiAddress,
             {
                mode: 'cors'
            },
            { headers: {
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Headers": "*"
            }}
        )
        .then(res => {
            if(res.status === 200){
                console.log('Server Online!!')
            }
            else{
                window.alert('Server temporarily offline for maintenance...')
            }
        }
        )  */
    }
    

    stringAdjust(originalString){
        var newString = originalString.split(',')[1]
        return newString
    }

    hitAPI(chosenImage){
        console.log("API hit!!")
        let apiFetchURL = config.apiAddress + 'predict';
        chosenImage = this.stringAdjust(chosenImage)
        /* console.log(chosenImage) */
        trackPromise(
            axios.post(
                apiFetchURL, 
                {
                    "chosenImage": chosenImage
                },
/*                  {
                    mode: 'cors'
                },
                { headers: {
                     "Access-Control-Allow-Origin": "*",
                     "Access-Control-Allow-Headers": "*"
                }} */
            ).then(res => {
                    this.setState({
                        inputTaken: true,
                        detectedImage: res.data.imstr,
                        status: res.data.success,
                        detections: res.data.detections
                    })                
                }
            ).catch(err => {
                    window.alert("Server is too busy right now or file is too big for it to handle. Please try again later." + err)
                    this.setState({
                        serverErr: true
                    })
                }
            )
        );
    }

    render() {
        return (
            <div className="mycss-disp">        
                <div id="mycss-area">
                    {this.state.serverErr
                    ?
                        <p><a href="/">Retry</a></p>
                    :
                        this.state.inputTaken
                        ?
                            <Output 
                                imstrProp={this.state.detectedImage} 
                                statusProp={this.state.status}
                                detectionsProp={this.state.detections}
                            />
                        :   
                            <div>
                                <Input hitAPIfunc={this.hitAPI} />
                            </div>
                        }
                </div>
            </div>
        )
    }
}



