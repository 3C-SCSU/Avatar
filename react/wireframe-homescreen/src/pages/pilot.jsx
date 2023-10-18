import React from "react";
import { Link } from 'react-router-dom';
import avatar_logo from '../assets/avatar_logo.png';
//this component is used for the pilot page

const Piloting = () => {
  return (
    <div>
        <Link to="/">
            <img style={{width: '75%', height: '150%', borderRadius: 4}} src={avatar_logo} alt='Avatar logo' />
        </Link>

      <h1>Brainwave Piloting</h1>
      <h3>Page for Brainwave Piloting Component</h3>
    </div>
  );
}


export default Piloting;