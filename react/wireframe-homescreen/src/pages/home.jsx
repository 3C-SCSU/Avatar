import React from 'react';
import avatar_logo from '../assets/avatar_logo.png'
import './home.css'
import brainwave_logo from '../assets/brainwave_piloting.png'
import drone_logo from '../assets/donation_logo.png'
import manual_logo from '../assets/manual_drone_control.png'
import { Link } from 'react-router-dom';

//home page
const Home = () => {
    return (
        <>
            <div className='top-section'>
                <div>
                    <img style={{width: '50%', height: '150%', borderRadius: 4}} src={avatar_logo} alt='Avatar logo' />
                    <h1>AVATAR</h1>
                </div>
            </div>
            
            <div className='bottom-section'>
                <Link to="/manualcontrol">
                    <button className='button-' style={{backgroundImage: `url(${brainwave_logo})`, backgroundSize:'cover', backgroundPosition: 'center', width: '400px', height: '350px', backgroundColor: '#75FEFE', marginRight: '20px', borderRadius: '10px', border: 'none', boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)'}}>Manual Drone Control</button>
                </Link>
                <Link to="/donation">
                    <button className='button-' style={{backgroundImage: `url(${drone_logo})`, backgroundSize: 'cover', backgroundPosition: 'center', width: '400px', height: '350px', backgroundColor: '#75FEFE', marginRight: '20px', borderRadius: '10px', border: 'none', boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)'}}>Brainwave Donation</button>
                </Link>
                <Link to="/pilot">
                    <button className='button-' style={{backgroundImage: `url(${manual_logo})`, backgroundSize: 'cover', backgroundPosition: 'center', width: '400px', height: '350px', backgroundColor: '#75FEFE', borderRadius: '10px', border: 'none', boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)'}}>Brainwave Piloting</button>
                </Link>
            </div>
        </>
    );
}

export default Home;