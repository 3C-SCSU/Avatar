import React from "react";
import { useState } from "react";
import './donation.css'
import { Link } from 'react-router-dom';
import avatar_logo from '../assets/avatar_logo.png'
//this component is used for the brainwave donation form page

const Donation = () => {
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    consent: false,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const inputValue = type === "checkbox" ? checked : value;

    setFormData({
      ...formData,
      [name]: inputValue,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission logic here
  };

  return (
    <div className="donation-form">
      <Link to="/">
        <img style={{width: '75%', height: '150%', borderRadius: 4}} src={avatar_logo} alt='Avatar logo' />
      </Link>
      <h1>Brainwave Donation</h1>
      <form className="donation-form-page" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="firstName">First Name</label>
          <input
            type="text"
            id="firstName"
            name="firstName"
            value={formData.firstName}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="lastName">Last Name</label>
          <input
            type="text"
            id="lastName"
            name="lastName"
            value={formData.lastName}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>
            <input
              type="checkbox"
              name="consent"
              checked={formData.consent}
              onChange={handleChange}
              required
            />
            I consent to donate my brainwaves.
          </label>
        </div>
        <div>
                  <button type="submit">Submit</button>
        <Link to="/">
          <button>Back</button>
        </Link>
        </div>

      </form>
    </div>
  );
};

export default Donation;