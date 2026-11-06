import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ThemeToggle from '../ThemeToggle';
import { ThemeProvider } from '../../contexts/ThemeContext';
import { BrandingManifest, EMPTY_MANIFEST } from '../../branding/manifest';

let mockManifest: BrandingManifest = EMPTY_MANIFEST;
jest.mock('../../contexts/BrandingContext', () => ({
  useBranding: () => ({ manifest: mockManifest, loaded: true }),
}));

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
    mockManifest = EMPTY_MANIFEST;
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

  it('offers the deployment themes and applies the chosen one', () => {
    mockManifest = { ...EMPTY_MANIFEST, themes: [{ id: 'acme', label: 'Acme' }] };
    renderToggle();
    choose('Acme');

    expect(document.documentElement.classList.contains('acme')).toBe(true);
  });

  it('hides the mode choices when branding locks the mode', () => {
    mockManifest = { ...EMPTY_MANIFEST, lockMode: 'light' };
    renderToggle();
    fireEvent.click(screen.getByRole('button', { name: 'Appearance' }));

    expect(screen.queryByRole('menuitem', { name: 'Dark' })).not.toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'High contrast' })).toBeInTheDocument();
  });
});
