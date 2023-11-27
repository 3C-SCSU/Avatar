import React from "react";
import { Link } from 'react-router-dom';
import avatar_logo from '../assets/avatar_logo.png';
//this component is used for manual dron control page

const ManualControl = () => {
  return (
    <div>
        <Link to="/">
            <img style={{width: '75%', height: '150%', borderRadius: 4}} src={avatar_logo} alt='Avatar logo' />
        </Link>
      <h1>Manual Drone Control</h1>
      <h3>Page for Manual Drone Control Component</h3>
    </div>
  );
}

export default ManualControl;