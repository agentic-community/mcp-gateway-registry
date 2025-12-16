import { Link } from 'react-router-dom';
import { KeyIcon, TicketIcon, ArrowRightIcon } from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';

// ============================================================================
// Credential Section Card
// ============================================================================

interface CredentialSectionProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  href: string;
  features: string[];
}

function CredentialSection({
  title,
  description,
  icon,
  href,
  features,
}: CredentialSectionProps) {
  return (
    <Link
      to={href}
      className="block bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-md transition-shadow p-6 group"
    >
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 group-hover:text-primary-600 dark:group-hover:text-primary-400">
              {title}
            </h3>
            <ArrowRightIcon className="w-5 h-5 text-gray-400 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors" />
          </div>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {description}
          </p>
          <ul className="mt-3 text-sm text-gray-600 dark:text-gray-400 space-y-1">
            {features.map((feature, index) => (
              <li key={index} className="flex items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 mr-2" />
                {feature}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Link>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function CredentialsPage() {
  return (
    <PageContent>
      <PageHeader
        title="Credentials"
        description="Manage API keys and gateway tokens for agent authentication"
        breadcrumbs={[{ name: 'Credentials' }]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <CredentialSection
          title="API Keys"
          description="Long-lived credentials for programmatic access"
          icon={<KeyIcon className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
          href="/credentials/api-keys"
          features={[
            'Create keys per agent',
            'Optional scope restrictions',
            'Optional expiration dates',
            'Revoke anytime',
          ]}
        />

        <CredentialSection
          title="Gateway Tokens"
          description="JWT tokens for MCP gateway authentication"
          icon={<TicketIcon className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
          href="/credentials/tokens"
          features={[
            'Mint tokens with custom TTL',
            'Scope-restricted tokens',
            'Revoke by JTI or bulk',
            'Decode tokens locally',
          ]}
        />
      </div>

      {/* Info section */}
      <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">
          Credential Types
        </h3>
        <div className="text-sm text-blue-700 dark:text-blue-400 space-y-2">
          <p>
            <strong>API Keys</strong> are long-lived secrets ideal for server-to-server
            communication. They are stored securely and can be revoked at any time.
          </p>
          <p>
            <strong>Gateway Tokens</strong> are short-lived JWTs that provide stateless
            authentication. They're ideal for distributed systems and can be scoped
            to specific permissions.
          </p>
        </div>
      </div>
    </PageContent>
  );
}
