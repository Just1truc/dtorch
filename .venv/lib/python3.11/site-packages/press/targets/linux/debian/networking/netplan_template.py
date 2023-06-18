NETPLAN_TEMPLATE = """#
# https://netplan.io/
#
network:
  version: 2
  renderer: networkd
  ethernets:
    {% for network in networks.networks %}
    {{ network.device }}:
      {% if network.interface == 'SNET' %}
      optional: true
      {% endif %}
      addresses: [ {{ network.ip_address }}/{{ network.prefix}} ]
      {% if network.gateway %}
      gateway4: {{ network.gateway }}
      {% endif %}
      {% if network.routes %}
      routes:
        {% for route in network.routes %}
        - to: {{ route.cidr }}
          via: {{ route.gateway }}
        {% endfor %}
      {% endif %}
      {% if network.interface == 'ExNET' %}
      nameservers:
        search: [{% for search in networks.dns.search %} {{ search }}{% endfor %}]
        addresses:
        {% for nameserver in networks.dns.nameservers %}
          - {{ nameserver }}
        {% endfor %}
      {% endif %}
    {% endfor %}
"""
