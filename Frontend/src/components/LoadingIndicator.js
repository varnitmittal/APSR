import React, { Component } from 'react'
import Loader from 'react-loader-spinner';
import { usePromiseTracker } from "react-promise-tracker";

const LoadingIndicator = props => {
    const { promiseInProgress } = usePromiseTracker();
    return (
        promiseInProgress &&

        <div
            style={{
                width: "100%",
                height: "100",
                display: "flex",
                justifyContent: "center",
                alignItems: "center"
            }}
            >
            <Loader type="Circles" color="#ffffff" height={100} width={100}/>
        </div>
    )
}

export default LoadingIndicator;