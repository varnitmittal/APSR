import React, { Component } from 'react';
import '../styling/css/sidebarfooter.css'

export default class SidebarFooter extends Component {
    render() {
        return (
            <div className='sbar-footer-container'>

                <div className="sbar-footer-begin">
                    <p>
                        Made with <i className="icon-heart" aria-hidden="true"></i> by 
                        <a href="https://github.com/varnitmittal/LPR" target="_blank"> Team HEX-CLAN</a>
                    </p>
                </div>

                <div className='sbar-footer-list'>
                    <div className="sbar-footer-list-item"><a href="#"><i className="icon-facebook2"></i></a></div>
                    <div className="sbar-footer-list-item"><a href="#"><i className="icon-twitter2"></i></a></div>
                    <div className="sbar-footer-list-item"><a href="#"><i className="icon-instagram"></i></a></div>
                    <div className="sbar-footer-list-item"><a href="#"><i className="icon-linkedin2"></i></a></div>
                </div>

                <p className='sbar-footer-end'>
                    &copy; hex-clan | 2021
                </p>

            </div>
        )
    }
}
