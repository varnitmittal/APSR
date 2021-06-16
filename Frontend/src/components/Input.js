import React, { Component } from 'react';
import '../styling/css/input.css';
import LoadingIndicator from './LoadingIndicator';

export default class Input extends Component {
    constructor(props){
        super(props);
        this.sendImage = this.sendImage.bind(this);
        this.onImageChange = this.onImageChange.bind(this);
        this.state = {
            buttonText: 'Upload an Image',
            isUp: false,
            isLoading:false
        }
    }
    
    onImageChange = (event) => {
        if (event.target.files && event.target.files[0]) {
          let reader = new FileReader();
          reader.onload = (e) => {
            this.setState({
                image: e.target.result,
                buttonText: 'Recognize Slots',
                isUp: true,
                isLoading: false
            });
          };
          reader.readAsDataURL(event.target.files[0]);
        }
    }

    sendImage = (e) => {
        this.props.hitAPIfunc(this.state.image)
        this.setState({
            isLoading: true
        })
    }

    
    render() {
        return (
            <div>
                {this.state.isLoading
                ?
                    <div>
                        <LoadingIndicator /><br />
                        <p style={{ fontSize: '1.3rem', color: '#ffffff' }}>This may take a few seconds, depending on image size. Please wait.</p>
                    </div>
                :
                    this.state.isUp
                        ?  
                            <div>
                                <div className="upload-btn-wrapper">
                                    <button type="submit" onClick={this.sendImage} className="ubtn">
                                        {this.state.buttonText}
                                    </button>
                                </div>
                                <div>
                                    <img id="target" style={{
                                            maxHeight:'50vh', 
                                            maxWidth:'80vw',
                                            minxHeight:'15vh', 
                                            minWidth:'15vw'}} 
                                        src={this.state.image}/>
                                </div>
                            </div>
                        :
                            <div className="upload-btn-wrapper">
                                <button type="button" className="ubtn">
                                    {this.state.buttonText}
                                    <input type="file" onChange={this.onImageChange}
                                    className="filetype" id="group_image" accept="image/*" 
                                />
                                </button>
                            </div>
                }
            </div>
        )
    }
}
