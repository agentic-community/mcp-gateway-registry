import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import FormField from '../FormField';
import TagsField from '../TagsField';
import StatusField from '../StatusField';
import VisibilityField from '../VisibilityField';
import MetadataField from '../MetadataField';
import AuthSchemeFields from '../AuthSchemeFields';
import ProxyField from '../ProxyField';
import { fieldClass } from '../formClasses';

describe('FormField', () => {
  it('renders the label and an asterisk when required', () => {
    render(
      <FormField label="Server Name" required>
        <input />
      </FormField>,
    );
    expect(screen.getByText(/Server Name/)).toHaveTextContent('Server Name *');
  });

  it('shows the error and not the hint when an error is present', () => {
    render(
      <FormField label="Name" error="Required" hint="some hint">
        <input />
      </FormField>,
    );
    expect(screen.getByText('Required')).toBeInTheDocument();
    expect(screen.queryByText('some hint')).not.toBeInTheDocument();
  });

  it('shows the hint when there is no error', () => {
    render(
      <FormField label="Name" hint="some hint">
        <input />
      </FormField>,
    );
    expect(screen.getByText('some hint')).toBeInTheDocument();
  });
});

describe('fieldClass', () => {
  it('defaults to the purple focus accent', () => {
    expect(fieldClass()).toContain('focus:ring-purple-500');
  });
  it('applies the requested accent', () => {
    expect(fieldClass('teal')).toContain('focus:ring-teal-500');
  });
  it('adds the error border when flagged', () => {
    expect(fieldClass('purple', true)).toContain('border-red-500');
  });
});

describe('TagsField', () => {
  it('joins the tag array for display', () => {
    render(<TagsField value={['a', 'b']} onChange={() => {}} />);
    expect(screen.getByDisplayValue('a,b')).toBeInTheDocument();
  });

  it('splits, trims, and drops empties on change', () => {
    const onChange = jest.fn();
    render(<TagsField value={[]} onChange={onChange} />);
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: ' a , b ,, c ' },
    });
    expect(onChange).toHaveBeenCalledWith(['a', 'b', 'c']);
  });
});

describe('StatusField', () => {
  it('reflects the value and reports changes', () => {
    const onChange = jest.fn();
    render(<StatusField value="active" onChange={onChange} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'deprecated' } });
    expect(onChange).toHaveBeenCalledWith('deprecated');
  });
});

describe('VisibilityField', () => {
  it('hides allowed-groups unless group-restricted', () => {
    const { rerender } = render(
      <VisibilityField
        value="public"
        onChange={() => {}}
        allowedGroups=""
        onAllowedGroupsChange={() => {}}
      />,
    );
    expect(screen.queryByText('Allowed Groups')).not.toBeInTheDocument();
    rerender(
      <VisibilityField
        value="group-restricted"
        onChange={() => {}}
        allowedGroups=""
        onAllowedGroupsChange={() => {}}
      />,
    );
    expect(screen.getByText('Allowed Groups')).toBeInTheDocument();
  });

  it('warns when group-restricted has no groups', () => {
    render(
      <VisibilityField
        value="group-restricted"
        onChange={() => {}}
        allowedGroups=""
        onAllowedGroupsChange={() => {}}
      />,
    );
    expect(
      screen.getByText(/At least one group is required/),
    ).toBeInTheDocument();
  });
});

describe('MetadataField', () => {
  it('renders a monospace textarea bound to value', () => {
    render(<MetadataField value='{"a":1}' onChange={() => {}} />);
    expect(screen.getByDisplayValue('{"a":1}')).toBeInTheDocument();
  });
});

describe('AuthSchemeFields', () => {
  const base = {
    credential: '',
    headerName: 'X-API-Key',
    onSchemeChange: jest.fn(),
    onCredentialChange: jest.fn(),
    onHeaderNameChange: jest.fn(),
  };

  it('shows neither credential nor header for none', () => {
    render(<AuthSchemeFields scheme="none" {...base} />);
    expect(document.querySelector('input[type="password"]')).not.toBeInTheDocument();
    expect(screen.queryByText('Header Name')).not.toBeInTheDocument();
  });

  it('shows the credential for bearer but not the header', () => {
    render(<AuthSchemeFields scheme="bearer" {...base} />);
    expect(document.querySelector('input[type="password"]')).toBeInTheDocument();
    expect(screen.queryByText('Header Name')).not.toBeInTheDocument();
  });

  it('shows credential and header for api_key', () => {
    render(<AuthSchemeFields scheme="api_key" {...base} />);
    expect(document.querySelector('input[type="password"]')).toBeInTheDocument();
    expect(screen.getByText('Header Name')).toBeInTheDocument();
  });
});

describe('ProxyField', () => {
  const base = {
    proxyTargetUrl: '',
    onIsProxiedChange: jest.fn(),
    onProxyTargetUrlChange: jest.fn(),
  };

  it('hides the target URL input until proxying is enabled', () => {
    const { rerender } = render(<ProxyField isProxied={false} {...base} />);
    expect(screen.queryByText('Proxy target URL')).not.toBeInTheDocument();
    rerender(<ProxyField isProxied={true} {...base} />);
    expect(screen.getByText('Proxy target URL')).toBeInTheDocument();
  });

  it('reports checkbox toggles through onIsProxiedChange', () => {
    const onIsProxiedChange = jest.fn();
    render(
      <ProxyField isProxied={false} {...base} onIsProxiedChange={onIsProxiedChange} />,
    );
    fireEvent.click(screen.getByRole('checkbox'));
    expect(onIsProxiedChange).toHaveBeenCalledWith(true);
  });

  it('warns about a missing target only when targetRequired and blank', () => {
    const { rerender } = render(
      <ProxyField isProxied={true} {...base} targetRequired />,
    );
    expect(
      screen.getByText(/A proxy target URL is required when proxying is enabled/),
    ).toBeInTheDocument();
    // A filled target clears the warning.
    rerender(
      <ProxyField
        isProxied={true}
        {...base}
        targetRequired
        proxyTargetUrl="https://backend.example.com/"
      />,
    );
    expect(
      screen.queryByText(/A proxy target URL is required when proxying is enabled/),
    ).not.toBeInTheDocument();
  });

  it('does not warn when the target is optional (native URL fallback)', () => {
    render(<ProxyField isProxied={true} {...base} />);
    expect(
      screen.queryByText(/A proxy target URL is required when proxying is enabled/),
    ).not.toBeInTheDocument();
  });

  it('surfaces an explicit validation error on the target field', () => {
    render(
      <ProxyField isProxied={true} {...base} targetRequired error="Bad URL" />,
    );
    expect(screen.getByText('Bad URL')).toBeInTheDocument();
  });
});
