import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ThemeToggle from '../ThemeToggle';
import { ThemeProvider } from '../../contexts/ThemeContext';

const renderToggle = () =>
  render(
    <ThemeProvider>
      <ThemeToggle />
    </ThemeProvider>,
  );

const choose = (label: string) => {
  fireEvent.click(screen.getByRole('button', { name: 'Appearance' }));
  fireEvent.click(screen.getByRole('menuitem', { name: label }));
};

describe('ThemeToggle', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.className = '';
    Reflect.deleteProperty(window, 'matchMedia');
  });

  it('applies and persists an explicit light choice', () => {
    // No matchMedia -> system resolves to dark.
    renderToggle();
    choose('Light');

    expect(document.documentElement.classList.contains('dark')).toBe(false);
    expect(localStorage.getItem('theme')).toBe('light');
  });

  it('returns to following the system and forgets the saved choice', () => {
    localStorage.setItem('theme', 'light');
    renderToggle();
    choose('System');

    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(localStorage.getItem('theme')).toBeNull();
  });

  it('toggles the high-contrast palette without changing the mode', () => {
    renderToggle();
    choose('High contrast');

    expect(document.documentElement.classList.contains('high-contrast')).toBe(true);
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });
});
