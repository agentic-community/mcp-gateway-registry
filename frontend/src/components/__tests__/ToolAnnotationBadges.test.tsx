/**
 * ToolAnnotationBadges shows READ ONLY and DESTRUCTIVE only for declared hints.
 */
import React from 'react';
import { render, screen } from '@testing-library/react';

import ToolAnnotationBadges from '../ToolAnnotationBadges';


describe('ToolAnnotationBadges', () => {
  it('shows READ ONLY for a read-only tool', () => {
    render(<ToolAnnotationBadges annotations={{ readOnlyHint: true }} />);

    expect(screen.getByText('READ ONLY')).toBeInTheDocument();
    expect(screen.queryByText('DESTRUCTIVE')).not.toBeInTheDocument();
  });

  it('shows DESTRUCTIVE for a destructive tool', () => {
    render(<ToolAnnotationBadges annotations={{ readOnlyHint: false, destructiveHint: true }} />);

    expect(screen.getByText('DESTRUCTIVE')).toBeInTheDocument();
  });

  it('never shows DESTRUCTIVE on a tool that claims read-only', () => {
    render(<ToolAnnotationBadges annotations={{ readOnlyHint: true, destructiveHint: true }} />);

    expect(screen.queryByText('DESTRUCTIVE')).not.toBeInTheDocument();
  });

  it('shows nothing without annotations or for undeclared hints', () => {
    const { container, rerender } = render(<ToolAnnotationBadges annotations={null} />);
    expect(container).toBeEmptyDOMElement();

    rerender(<ToolAnnotationBadges annotations={{ readOnlyHint: false }} />);
    expect(container).toBeEmptyDOMElement();
  });
});
