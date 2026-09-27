import React from 'react';

export function Card({ title, subtitle, icon: Icon, children, rightAction }) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            {Icon && <Icon size={16} color="var(--quantum-violet)" />}
            <span>{title}</span>
          </div>
          {subtitle && <div className="card-subtitle">{subtitle}</div>}
        </div>
        {rightAction && <div>{rightAction}</div>}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}
